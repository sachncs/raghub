"""``raghub run`` — serve the FastAPI app under uvicorn.

Registers the ``run`` command on the main Typer app via :func:`register`.
"""

from __future__ import annotations

import typer
import uvicorn


def register(app: "typer.Typer") -> None:
    """Attach the ``run`` command to ``app``."""

    @app.command(name="run")
    def run(
        host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind host."),
        port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
        workers: int = typer.Option(1, "--workers", "-w", help="uvicorn worker count."),
        reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes."),
    ) -> None:
        """Run the FastAPI app under uvicorn (foreground).

        Production deployments should set ``--workers`` > 1; ``--reload`` is for development.
        """
        config = uvicorn.Config(
            "raghub.api.app:create_app",
            factory=True,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
        )
        uvicorn.Server(config).run()


