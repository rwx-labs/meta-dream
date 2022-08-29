//! Database interface

use sqlx::{migrate::Migrator, AnyPool};

use crate::Error;

static MIGRATOR: Migrator = sqlx::migrate!();

pub async fn open(url: &str) -> Result<AnyPool, Error> {
    let pool = AnyPool::connect(url)
        .await
        .map_err(Error::DatabaseOpenError)?;

    Ok(pool)
}

pub async fn migrate(pool: AnyPool) -> Result<(), Error> {
    MIGRATOR
        .run(&pool)
        .await
        .map_err(Error::DatabaseMigrationError)
}
