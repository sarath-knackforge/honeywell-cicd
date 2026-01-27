from mlflow import MlflowClient
from loguru import logger

def promote_challenger_to_champion(
    model_name: str,
    allow_promotion: bool,
) -> None:
    """
    Swap aliases so that:
      - Challenger → Champion
      - Old Champion → Challenger

    This function will REFUSE to run unless allow_promotion=True.
    """

    if not allow_promotion:
        raise RuntimeError(
            "Alias swap blocked: promotion_gate did not approve promotion."
        )

    client = MlflowClient()

    challenger_mv = client.get_model_version_by_alias(model_name, "Challenger")

    try:
        champion_mv = client.get_model_version_by_alias(model_name, "Champion")
        old_champion_version = champion_mv.version
    except Exception:
        old_champion_version = None

    new_champion_version = challenger_mv.version

    # Promote Challenger → Champion
    client.set_registered_model_alias(
        name=model_name,
        alias="Champion",
        version=new_champion_version,
    )

    # Demote old Champion → Challenger
    if old_champion_version is not None:
        client.set_registered_model_alias(
            name=model_name,
            alias="Challenger",
            version=old_champion_version,
        )

    logger.info(
        "Alias swap complete → Champion=%s Challenger=%s",
        new_champion_version,
        old_champion_version,
    )
