import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

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
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sec))

class RawCall(BaseModel):
    method: str = Field(default="POST", description="HTTP method: GET/POST")
    path: str = Field(description="API path, e.g. /API/System/DeviceInfo/Get")
    json: Optional[Dict[str, Any]] = Field(default=None, description="JSON body for POST requests")

@dataclass
class AddonConfig:
    nvr_host: str
    nvr_port: int
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
    def __init__(self, cfg: AddonConfig):
        scheme = "https" if cfg.https else "http"
        self.base = f"{scheme}://{cfg.nvr_host}:{cfg.nvr_port}"
        self.cfg = cfg
        self.client = httpx.AsyncClient(base_url=self.base, timeout=20.0)
        self.logged_in = False
        self.session_info: Dict[str, Any] = {}

    async def close(self):
        await self.client.aclose()

    async def call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        method = method.upper().strip()
        if not path.startswith("/"):
            path = "/" + path

        if method == "GET":
            r = await self.client.get(path)
        else:
            r = await self.client.post(path, json=body or {})
        try:
            data = r.json()
        except Exception:
            raise HTTPException(status_code=502, detail=f"NVR non-JSON response ({r.status_code}): {r.text[:300]}")

        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail={"status": r.status_code, "data": data})

        return data

    async def login(self) -> None:
        # Optional: Get info before login
        try:
            await self.call("POST", "/API/Login/Range", {"Action": "Get"})
        except Exception:
            pass

        payload_variants = [
            {"UserName": self.cfg.username, "Password": self.cfg.password},
            {"User": self.cfg.username, "PassWord": self.cfg.password},
            {"Username": self.cfg.username, "Password": self.cfg.password},
        ]
        last_err = None
        for p in payload_variants:
            try:
                resp = await self.call("POST", "/API/Login/Login", {"Action": "Login", **p})
                self.logged_in = True
                self.session_info = resp
                return
            except Exception as e:
                last_err = e
        self.logged_in = False
        if last_err:
            raise last_err

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
                print("FACE:", json.dumps(payload, ensure_ascii=False)[:2000])

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
                print("PLATE:", json.dumps(payload, ensure_ascii=False)[:2000])

            await self.ha.fire_event(f"{self.cfg.event_prefix}_plate", payload)
            await self.ha.set_state(
                f"{self.cfg.state_prefix}_plate",
                plate or "unknown",
                {"uid": uid, "last_update": _ts_iso(time.time()), "source": "raysharp_nvr_ai_bridge"},
            )

    async def _poll_faces(self, start_s: float, end_s: float) -> None:
        body: Dict[str, Any] = {
            "Action": "Search",
            "StartTime": _ts_iso(start_s),
            "EndTime": _ts_iso(end_s),
        }
        if self.cfg.channels:
            body["Chn"] = self.cfg.channels
        if self.cfg.faces_alarm_groups:
            body["AlarmGroup"] = self.cfg.faces_alarm_groups
        if self.cfg.faces_similarity_min:
            body["Similarity"] = self.cfg.faces_similarity_min

        resp = await self.nvr.call("POST", "/API/AI/SnapedFaces", body)
        search_id = resp.get("SearchID") or resp.get("SearchId") or resp.get("ID") or resp.get("Id")
        if not search_id:
            items = resp.get("Faces") or resp.get("Items") or resp.get("Data") or []
            await self._emit_faces(items)
            return

        idx = 0
        while True:
            page = await self.nvr.call("POST", "/API/AI/SnapedFaces", {"Action": "GetByIndex", "SearchID": search_id, "Index": idx})
            items = page.get("Faces") or page.get("Items") or page.get("Data") or []
            if not items:
                break
            await self._emit_faces(items)
            idx += len(items)

        await self.nvr.call("POST", "/API/AI/SnapedFaces", {"Action": "StopSearch", "SearchID": search_id})

    async def _poll_plates(self, start_s: float, end_s: float) -> None:
        body: Dict[str, Any] = {
            "Action": "SearchPlate",
            "StartTime": _ts_iso(start_s),
            "EndTime": _ts_iso(end_s),
        }
        if self.cfg.channels:
            body["Chn"] = self.cfg.channels
        if self.cfg.plates_alarm_groups:
            body["AlarmGroup"] = self.cfg.plates_alarm_groups
        body["MaxErrorCharCnt"] = self.cfg.plates_max_error_char_cnt

        resp = await self.nvr.call("POST", "/API/AI/SnapedObjects", body)
        search_id = resp.get("SearchID") or resp.get("SearchId") or resp.get("ID") or resp.get("Id")
        if not search_id:
            items = resp.get("Plates") or resp.get("Items") or resp.get("Data") or []
            await self._emit_plates(items)
            return

        idx = 0
        while True:
            page = await self.nvr.call("POST", "/API/AI/SnapedObjects", {"Action": "GetByIndex", "SearchID": search_id, "Index": idx})
            items = page.get("Plates") or page.get("Items") or page.get("Data") or []
            if not items:
                break
            await self._emit_plates(items)
            idx += len(items)

        await self.nvr.call("POST", "/API/AI/SnapedObjects", {"Action": "StopSearch", "SearchID": search_id})

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
                print(f"[bridge] poll error: {e}")
                self.nvr.logged_in = False
                await asyncio.sleep(min(10, self.cfg.poll_interval_s))
            finally:
                await asyncio.sleep(self.cfg.poll_interval_s)

app = FastAPI(title="Raysharp NVR AI Bridge", version="0.1.0")

CFG = None
NVR = None
HA = None
ENGINE = None

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
        return await NVR.call(call.method, call.path, call.json)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
