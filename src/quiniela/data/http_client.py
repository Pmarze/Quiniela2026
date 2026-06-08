from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    content_type: str
    body: bytes


def fetch_url(url: str, timeout_seconds: int = 30) -> HttpResponse:
    request = Request(
        url,
        headers={
            "User-Agent": "quiniela2026-data-pipeline/0.1",
            "Accept": "application/json,text/csv,text/plain,*/*",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            status = int(getattr(response, "status", 200))
            content_type = response.headers.get("Content-Type", "")
            return HttpResponse(url=url, status=status, content_type=content_type, body=body)
    except HTTPError as exc:
        body = exc.read()
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return HttpResponse(url=url, status=int(exc.code), content_type=content_type, body=body)
    except URLError as exc:
        raise RuntimeError(f"No se pudo descargar {url}: {exc}") from exc

