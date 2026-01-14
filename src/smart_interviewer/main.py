import logging
import uvicorn
from smart_interviewer.settings import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)




def main() -> None:
    host = settings.HOST
    port = settings.PORT
    reload = settings.RELOAD


    uvicorn.run(
        "smart_interviewer.api:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()
