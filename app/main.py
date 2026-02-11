import asyncio
import json
import time
import gzip
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Union

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SUPERVISOR_TOKEN = None
try:
    import os
    SUPERVISOR_TOKEN = os.environ.get("SUPERVISOR_TOKEN")
except Exception:
    SUPERVISOR_TOKEN = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _ts_iso(sec: float) -> str:
    # NVR API typically expects "YYYY-MM-DD HH:MM:SS"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sec))


def _cache_buster() -> str:
    # UI uses "YYYY-MM-DD@HH:MM:SS" as query cache buster
    return time.strftime("%Y-%m-%d@%H:%M:%S", time.localtime())


class RawCall(BaseModel):
    method: str = Field(default="POST", description="HTTP method: GET/POST")
    path: str = Field(description="API path, e.g. /API/AccountRules/Get")
    body: Optional[Dict[str, Any]] = Field(default=None, description="JSON body for POST requests")


@dataclass
class AddonConfig:
    nvr_host: str
    nvr_port: int
    api_prefix: str
    cache_bust: bool
    https: bool
    username: str
    password: str
    poll_interval_s: int
    lookback_s: int
    channels: List[int]

    ai_enable: bool
    faces_enable: bool
    faces_alarm_groups: List[int]
    faces_similarity_min: int

    plates_enable: bool
    plates_alarm_groups: List[int]
    plates_max_error_char_cnt: int

    publish_to_ha: bool
    event_prefix: str
    publish_states: bool
    state_prefix: str
    debug_log_payloads: bool


def load_addon_config() -> AddonConfig:
    with open("/data/options.json", "r", encoding="utf-8") as f:
        opt = json.load(f)

    ai = opt.get("ai", {}) or {}
    faces = ai.get("faces", {}) or {}
    plates = ai.get("plates", {}) or {}
    events = opt.get("events", {}) or {}

    return AddonConfig(
        nvr_host=opt["nvr_host"],
        nvr_port=int(opt.get("nvr_port", 80)),
        api_prefix=str(opt.get("api_prefix", "/API")),
        cache_bust=bool(opt.get("cache_bust", True)),
        https=bool(opt.get("https", False)),
        username=opt.get("username", "admin"),
        password=opt.get("password", ""),
        poll_interval_s=int(opt.get("poll_interval_s", 5)),
        lookback_s=int(opt.get("lookback_s", 20)),
        channels=[int(x) for x in (opt.get("channels") or [])],

        ai_enable=bool(ai.get("enable", True)),
        faces_enable=bool(faces.get("enable", True)),
        faces_alarm_groups=[int(x) for x in (faces.get("alarm_groups") or [])],
        faces_similarity_min=int(faces.get("similarity_min", 70)),

        plates_enable=bool(plates.get("enable", True)),
        plates_alarm_groups=[int(x) for x in (plates.get("alarm_groups") or [])],
        plates_max_error_char_cnt=int(plates.get("max_error_char_cnt", 0)),

        publish_to_ha=bool(events.get("publish_to_ha", True)),
        event_prefix=str(events.get("event_prefix", "raysharp_nvr")),
        publish_states=bool(events.get("publish_states", True)),
        state_prefix=str(events.get("state_prefix", "sensor.raysharp_nvr_last")),
        debug_log_payloads=bool(opt.get("debug_log_payloads", False)),
    )


class NvrClient:
    """
    API format confirmed by browser (.har):
      POST http://<nvr>/API/<Module>/<Action>?YYYY-MM-DD@HH:MM:SS

    Many API endpoints expect:
      - JSON wrapper: {"version":"1.0","data":{...}}
      - gzip body + header Content-Encoding: gzip

    Login confirmed:
      POST /API/Web/Login
      (browser sends plain JSON, not gzip)
      Digest auth challenge + Set-Cookie session_<port> + X-CsrfToken
    """

    def __init__(self, cfg: AddonConfig):
        scheme = "https" if cfg.https else "http"
        self.base = f"{scheme}://{cfg.nvr_host}:{cfg.nvr_port}"
        self.cfg = cfg

        self._auth = httpx.DigestAuth(cfg.username, cfg.password)
        self.client = httpx.AsyncClient(
            base_url=self.base,
            timeout=20.0,
            auth=self._auth,
            headers={"Accept": "application/json"},
        )

        self.logged_in = False
        self.session_cookie_name: Optional[str] = None  # e.g. session_80
        self.csrf_token: Optional[str] = None

    async def close(self):
        await self.client.aclose()

    def _normalize_path(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path

        # keep explicit /API or /api
        if path.startswith("/API/") or path.startswith("/api/"):
            return path

        # relative -> prefix
        return self.cfg.api_prefix.rstrip("/") + path

    def _with_cache_bust(self, path: str) -> str:
        if not self.cfg.cache_bust:
            return path
        sep = "&" if "?" in path else "?"
        return f"{path}{sep}{_cache_buster()}"

    @staticmethod
    def _wrap_version_data(body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        If already wrapped, keep as-is.
        Otherwise wrap into {"version":"1.0","data": body}.
        """
        body = body or {}
        if isinstance(body, dict) and "version" in body and "data" in body:
            return body
        return {"version": "1.0", "data": body}

    @staticmethod
    def _gzip_json(payload: Dict[str, Any]) -> bytes:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return gzip.compress(raw)

    async def call(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        *,
        gzip_body: bool = True,
        wrap_version: bool = True,
    ) -> Dict[str, Any]:
        method = method.upper().strip()
        path = self._normalize_path(path)
        url = self._with_cache_bust(path)

        headers: Dict[str, str] = {}

        # CSRF token (browser sends X-csrftoken)
        if self.csrf_token:
            headers["X-csrftoken"] = self.csrf_token

        if method == "GET":
            r = await self.client.get(url, headers=headers)
        else:
            # For most API endpoints we mimic browser: wrap + gzip + Content-Encoding
            payload = body or {}
            if wrap_version:
                payload = self._wrap_version_data(payload)

            if gzip_body:
                content = self._gzip_json(payload)
                headers["Content-Encoding"] = "gzip"
                headers["Content-Type"] = "application/json; charset=utf-8"
                r = await self.client.post(url, content=content, headers=headers)
            else:
                headers["Content-Type"] = "application/json"
                r = await self.client.post(url, json=payload, headers=headers)

        # Parse response
        try:
            data = r.json()
        except Exception:
            data = {"_non_json": True, "status": r.status_code, "text": r.text[:4000]}

        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail={"status": r.status_code, "data": data})

        return data

    async def login(self) -> None:
        """
        Browser login:
          POST /API/Web/Login
          Content-type: application/json (NOT gzip)
        """
        url = self._with_cache_bust(f"{self.cfg.api_prefix}/Web/Login")

        # IMPORTANT: do NOT wrap/gzip login, mimic browser.
        r = await self.client.post(url, json={}, headers={"Content-Type": "application/json", "Accept": "application/json"})

        # CSRF token header
        csrf = r.headers.get("X-CsrfToken") or r.headers.get("x-csrftoken") or r.headers.get("X-csrftoken")
        if csrf:
            self.csrf_token = csrf

        # session cookie session_<port>
        self.session_cookie_name = None
        for c in self.client.cookies.jar:
            if c.name.startswith("session_"):
                self.session_cookie_name = c.name
                break

        if r.status_code >= 400:
            try:
                data = r.json()
            except Exception:
                data = {"text": r.text[:4000]}
            self.logged_in = False
            raise HTTPException(status_code=502, detail={"status": r.status_code, "data": data})

        self.logged_in = True


class HaPublisher:
    def __init__(self, cfg: AddonConfig):
        self.cfg = cfg
        self.client = httpx.AsyncClient(timeout=15.0)

    async def close(self):
        await self.client.aclose()

    def _headers(self) -> Dict[str, str]:
        if not SUPERVISOR_TOKEN:
            return {}
        return {"Authorization": f"Bearer {SUPERVISOR_TOKEN}", "Content-Type": "application/json"}

    async def fire_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.cfg.publish_to_ha or not SUPERVISOR_TOKEN:
            return
        url = f"http://supervisor/core/api/events/{event_type}"
        await self.client.post(url, headers=self._headers(), json=payload)

    async def set_state(self, entity_id: str, state: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        if not self.cfg.publish_states or not SUPERVISOR_TOKEN:
            return
        url = f"http://supervisor/core/api/states/{entity_id}"
        body = {"state": str(state), "attributes": attributes or {}}
        await self.client.post(url, headers=self._headers(), json=body)


class PollEngine:
    def __init__(self, cfg: AddonConfig, nvr: NvrClient, ha: HaPublisher):
        self.cfg = cfg
        self.nvr = nvr
        self.ha = ha
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._seen_faces: Dict[str, int] = {}
        self._seen_plates: Dict[str, int] = {}

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    def _cache_put(self, cache: Dict[str, int], key: str):
        now = _now_ms()
        cache[key] = now
        if len(cache) > 5000:
            for k, _ in sorted(cache.items(), key=lambda kv: kv[1])[:1000]:
                cache.pop(k, None)

    def _cache_has(self, cache: Dict[str, int], key: str) -> bool:
        return key in cache

    async def _emit_faces(self, items: List[Dict[str, Any]]) -> None:
        for it in items:
            uid = str(it.get("UUId") or it.get("UUID") or it.get("Id") or it.get("ID") or "")
            if not uid:
                uid = json.dumps(it, sort_keys=True)[:180]
            if self._cache_has(self._seen_faces, uid):
                continue
            self._cache_put(self._seen_faces, uid)

            payload = {"type": "face", "uid": uid, "data": it}
            if self.cfg.debug_log_payloads:
                print("FACE:", json.dumps(payload, ensure_ascii=False)[:4000])

            await self.ha.fire_event(f"{self.cfg.event_prefix}_face", payload)
            await self.ha.set_state(
                f"{self.cfg.state_prefix}_face_uid",
                uid,
                {"last_update": _ts_iso(time.time()), "source": "raysharp_nvr_ai_bridge"},
            )

    async def _emit_plates(self, items: List[Dict[str, Any]]) -> None:
        for it in items:
            uid = str(it.get("UUId") or it.get("UUID") or it.get("Id") or it.get("ID") or it.get("PlateID") or "")
            if not uid:
                uid = json.dumps(it, sort_keys=True)[:180]
            if self._cache_has(self._seen_plates, uid):
                continue
            self._cache_put(self._seen_plates, uid)

            plate = it.get("Plate") or it.get("PlateNo") or it.get("License") or ""
            payload = {"type": "plate", "uid": uid, "plate": plate, "data": it}
            if self.cfg.debug_log_payloads:
                print("PLATE:", json.dumps(payload, ensure_ascii=False)[:4000])

            await self.ha.fire_event(f"{self.cfg.event_prefix}_plate", payload)
            await self.ha.set_state(
                f"{self.cfg.state_prefix}_plate",
                plate or "unknown",
                {"uid": uid, "last_update": _ts_iso(time.time()), "source": "raysharp_nvr_ai_bridge"},
            )

    async def _poll_faces(self, start_s: float, end_s: float) -> None:
        # IMPORTANT: For this OEM most endpoints want {"version":"1.0","data":{...}} + gzip.
        # Our NvrClient.call already wraps+gzips by default.
        data_body: Dict[str, Any] = {
            "StartTime": _ts_iso(start_s),
            "EndTime": _ts_iso(end_s),
        }

        if self.cfg.channels:
            # OEMs vary; keep list if multiple, int if single
            if len(self.cfg.channels) == 1:
                data_body["Chn"] = int(self.cfg.channels[0])
            else:
                data_body["Chn"] = [int(x) for x in self.cfg.channels]

        if self.cfg.faces_alarm_groups:
            data_body["AlarmGroup"] = [int(x) for x in self.cfg.faces_alarm_groups]

        if self.cfg.faces_similarity_min is not None:
            data_body["Similarity"] = int(self.cfg.faces_similarity_min)

        # Action in URL (confirmed)
        resp = await self.nvr.call("POST", "/API/AI/SnapedFaces/Search", data_body, gzip_body=True, wrap_version=True)

        # Many endpoints return {"data":{...}}
        data = resp.get("data", resp)

        search_id = data.get("SearchID") or data.get("SearchId") or data.get("ID") or data.get("Id")
        if not search_id:
            items = data.get("Faces") or data.get("Items") or data.get("Data") or []
            await self._emit_faces(items)
            return

        idx = 0
        while True:
            page = await self.nvr.call(
                "POST",
                "/API/AI/SnapedFaces/GetByIndex",
                {"SearchID": search_id, "Index": idx},
                gzip_body=True,
                wrap_version=True,
            )
            pdata = page.get("data", page)
            items = pdata.get("Faces") or pdata.get("Items") or pdata.get("Data") or []
            if not items:
                break
            await self._emit_faces(items)
            idx += len(items)

        await self.nvr.call("POST", "/API/AI/SnapedFaces/StopSearch", {"SearchID": search_id}, gzip_body=True, wrap_version=True)

    async def _poll_plates(self, start_s: float, end_s: float) -> None:
        data_body: Dict[str, Any] = {
            "StartTime": _ts_iso(start_s),
            "EndTime": _ts_iso(end_s),
            "MaxErrorCharCnt": int(self.cfg.plates_max_error_char_cnt),
        }

        if self.cfg.channels:
            if len(self.cfg.channels) == 1:
                data_body["Chn"] = int(self.cfg.channels[0])
            else:
                data_body["Chn"] = [int(x) for x in self.cfg.channels]

        if self.cfg.plates_alarm_groups:
            data_body["AlarmGroup"] = [int(x) for x in self.cfg.plates_alarm_groups]

        resp = await self.nvr.call("POST", "/API/AI/SnapedObjects/SearchPlate", data_body, gzip_body=True, wrap_version=True)
        data = resp.get("data", resp)

        search_id = data.get("SearchID") or data.get("SearchId") or data.get("ID") or data.get("Id")
        if not search_id:
            items = data.get("Plates") or data.get("Items") or data.get("Data") or []
            await self._emit_plates(items)
            return

        idx = 0
        while True:
            page = await self.nvr.call(
                "POST",
                "/API/AI/SnapedObjects/GetByIndex",
                {"SearchID": search_id, "Index": idx},
                gzip_body=True,
                wrap_version=True,
            )
            pdata = page.get("data", page)
            items = pdata.get("Plates") or pdata.get("Items") or pdata.get("Data") or []
            if not items:
                break
            await self._emit_plates(items)
            idx += len(items)

        await self.nvr.call("POST", "/API/AI/SnapedObjects/StopSearch", {"SearchID": search_id}, gzip_body=True, wrap_version=True)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.nvr.logged_in:
                    await self.nvr.login()

                end_s = time.time()
                start_s = end_s - float(self.cfg.lookback_s)

                if self.cfg.ai_enable and self.cfg.faces_enable:
                    await self._poll_faces(start_s, end_s)
                if self.cfg.ai_enable and self.cfg.plates_enable:
                    await self._poll_plates(start_s, end_s)

            except Exception as e:
                import traceback
                print("[bridge] poll error:", repr(e))
                traceback.print_exc()
                self.nvr.logged_in = False
                await asyncio.sleep(min(10, self.cfg.poll_interval_s))
            finally:
                await asyncio.sleep(self.cfg.poll_interval_s)


app = FastAPI(title="Raysharp NVR AI Bridge", version="0.1.0")

CFG: Optional[AddonConfig] = None
NVR: Optional[NvrClient] = None
HA: Optional[HaPublisher] = None
ENGINE: Optional[PollEngine] = None


@app.on_event("startup")
async def startup():
    global CFG, NVR, HA, ENGINE
    CFG = load_addon_config()
    NVR = NvrClient(CFG)
    HA = HaPublisher(CFG)
    ENGINE = PollEngine(CFG, NVR, HA)
    ENGINE.start()


@app.on_event("shutdown")
async def shutdown():
    global NVR, HA, ENGINE
    if ENGINE:
        await ENGINE.stop()
    if NVR:
        await NVR.close()
    if HA:
        await HA.close()


@app.get("/health")
async def health():
    return {"ok": True, "logged_in": bool(getattr(NVR, "logged_in", False))}


@app.post("/api/raw")
async def api_raw(call: RawCall):
    if not NVR:
        raise HTTPException(status_code=503, detail="Not initialized")
    try:
        if not NVR.logged_in:
            await NVR.login()

        # For RAW: default to "API style" (wrap+gzip) unless user explicitly calls /API/Web/Login etc.
        # If you want plain json, call path under /API/Web/... and pass body; we will still wrap by default,
        # but you can force plain by providing body={"version":"1.0","data":...} and it will pass as-is.
        return await NVR.call(call.method, call.path, call.body)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=repr(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
