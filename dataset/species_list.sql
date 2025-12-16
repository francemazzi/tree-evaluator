-- Species List (speciesList.csv) - SQLite schema
-- Source CSV columns:
-- Genus Name, Species Name, Synonyms, Family, Order, Class, Common Name, Species Code,
-- Growth Form, Percent Leaf Type, Leaf Type, Growth Rate, Longevity, Height at Maturity (feet)

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

DROP TABLE IF EXISTS species_list;

CREATE TABLE species_list (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genus_name TEXT NOT NULL,
    species_name TEXT NOT NULL,
    genus_species TEXT NOT NULL,
    synonyms TEXT,
    family TEXT,
    taxonomic_order TEXT,
    taxonomic_class TEXT,
    common_name TEXT,
    species_code TEXT,
    growth_form TEXT,
    percent_leaf_type TEXT,
    leaf_type TEXT,
    growth_rate TEXT,
    longevity TEXT,
    height_at_maturity_feet INTEGER,
    source_row_hash TEXT
);

-- Helpful indexes for lookup / autocomplete style queries
CREATE INDEX IF NOT EXISTS idx_species_list_genus_name ON species_list(genus_name);
CREATE INDEX IF NOT EXISTS idx_species_list_species_name ON species_list(species_name);
CREATE INDEX IF NOT EXISTS idx_species_list_genus_species ON species_list(genus_species);
CREATE INDEX IF NOT EXISTS idx_species_list_family ON species_list(family);

-- Prefer unique codes when present
CREATE UNIQUE INDEX IF NOT EXISTS idx_species_list_species_code_unique
ON species_list(species_code)
WHERE species_code IS NOT NULL AND species_code <> '';


