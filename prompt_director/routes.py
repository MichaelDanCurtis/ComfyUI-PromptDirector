from __future__ import annotations

from typing import Any

from .contracts import ContractValidationError, load_contract_schema
from .credentials import (
    CredentialUnavailable,
    delete_provider_profile,
    list_provider_profiles,
    load_provider_credentials,
    save_provider_profile,
)
from .inspection import list_inspections, list_inspections_v1
from .manifest import module_manifest


def register_routes() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:
        return False

    routes = PromptServer.instance.routes

    @routes.get("/prompt_director/credentials")
    async def get_credentials(_request):
        return web.json_response({"ok": True, **list_provider_profiles()})

    @routes.post("/prompt_director/credentials")
    async def set_credentials(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise CredentialUnavailable("JSON object required")
            result = save_provider_profile(
                profile_name=str(body.get("profile") or "default"),
                provider_name=str(body.get("provider") or ""),
                values={
                    key: body[key]
                    for key in ("api_key", "base_url", "model", "auth_style", "headers")
                    if key in body
                },
            )
            return web.json_response({"ok": True, **result})
        except (CredentialUnavailable, ValueError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @routes.delete("/prompt_director/credentials")
    async def clear_credentials(request):
        try:
            body = await request.json()
            result = delete_provider_profile(
                profile_name=str(body.get("profile") or "default"),
                provider_name=str(body.get("provider") or ""),
            )
            return web.json_response({"ok": True, **result})
        except (CredentialUnavailable, ValueError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @routes.post("/prompt_director/credentials/check")
    async def check_credentials(request):
        try:
            body = await request.json()
            profile = str(body.get("profile") or "default")
            provider = str(body.get("provider") or "")
            resolved = load_provider_credentials(provider_name=provider, profile_name=profile)
            return web.json_response(
                {
                    "ok": True,
                    "profile": profile,
                    "provider": provider,
                    "configured": True,
                    "base_url": resolved.get("base_url", ""),
                    "model": resolved.get("model", ""),
                    "auth_style": resolved.get("auth_style", "bearer"),
                    "header_names": sorted((resolved.get("headers") or {}).keys()),
                }
            )
        except CredentialUnavailable as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    @routes.get("/prompt_director/inspection")
    async def get_inspection(request):
        return web.json_response(
            {
                "ok": True,
                **list_inspections(node_id=request.query.get("node_id")),
            }
        )

    @routes.get("/prompt_director/v1/manifest")
    async def get_manifest(_request):
        return web.json_response({"ok": True, **module_manifest()})

    @routes.get("/prompt_director/v1/contracts/{contract_name}")
    async def get_contract_schema(request):
        try:
            return web.json_response(
                {
                    "ok": True,
                    "contract_name": request.match_info["contract_name"],
                    "schema": load_contract_schema(request.match_info["contract_name"]),
                }
            )
        except ContractValidationError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)

    @routes.get("/prompt_director/v1/inspections")
    async def get_inspections_v1(request):
        return web.json_response(
            {
                "ok": True,
                **list_inspections_v1(node_id=request.query.get("node_id")),
            }
        )

    @routes.get("/prompt_director/v1/inspections/{node_id}")
    async def get_inspection_v1(request):
        return web.json_response(
            {
                "ok": True,
                **list_inspections_v1(node_id=request.match_info["node_id"]),
            }
        )

    return True


ROUTES_REGISTERED = register_routes()


__all__ = ["ROUTES_REGISTERED", "register_routes"]
