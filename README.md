# Raysharp NVR AI Bridge (Home Assistant Add-on)

## Концепция: “включить всё, что позволяет API”
1) Встроенный poller уже тянет AI-события (лица/номера) и публикует их в HA как события.
2) Для *любых других* функций API (диски, тревоги, сеть, системные параметры и т.д.) есть универсальный прокси:
   `POST http://<HA>:8000/api/raw` → вызывает любой `/API/...` endpoint.

Таким образом, аддон “покрывает” **всю** документацию сразу:
- то, что уже умеем опрашивать — делаем автоматом (AI),
- то, что вы хотите добавить следующим — включается без перепаковки, через `/api/raw`,
  а потом можно оформить как отдельный poller/сенсоры.

## События в Home Assistant
- `event_type: <event_prefix>_face`
- `event_type: <event_prefix>_plate`

## Пример автоматизации (номера)
```yaml
trigger:
  - platform: event
    event_type: raysharp_nvr_plate
action:
  - service: logbook.log
    data:
      name: NVR
      message: "Plate: {{ trigger.event.data.plate }}"
```

## Диагностика
- `GET /health`
- `POST /api/raw`
  ```json
  {"method":"POST","path":"/API/System/DeviceInfo/Get","json":{"Action":"Get"}}
  ```

## Важное по логину
В некоторых прошивках пароль передаётся через TransKey/PBKDF2 (RSA + salt), а в HTML иногда нет примеров ответа.
В этом первом релизе реализован best-effort "plain" login.
Если ваш NVR требует TransKey — включите `debug_log_payloads: true` и пришлите фрагмент ответа на логин/ошибку:
по нему добавим точный алгоритм.
