-- Remove tables that are not needed
DROP TABLE IF EXISTS photometry_skipped;
DROP TABLE IF EXISTS datavalidation;
DROP TABLE IF EXISTS datavalidation_corr;
DROP TABLE IF EXISTS diagnostics_corr;

-- Only keep targets from a few CCDs
DELETE FROM todolist WHERE camera != 1 OR ccd IN (2,3);

-- Optimize tables
VACUUM;
ANALYZE;
VACUUM;