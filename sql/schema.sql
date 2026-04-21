
-- patents table
CREATE TABLE patents (
    patent_id    TEXT PRIMARY KEY,
    patent_type  TEXT,
    filing_date  TEXT,
    year         INTEGER,
    title        TEXT,
    abstract     TEXT
);

-- inventors table
CREATE TABLE inventors (
    inventor_id  TEXT,
    patent_id    TEXT,
    full_name    TEXT,
    country      TEXT
);

-- companies table
CREATE TABLE companies (
    company_id    TEXT,
    patent_id     TEXT,
    name          TEXT,
    assignee_type REAL
);

-- relationships table
CREATE TABLE relationships (
    patent_id   TEXT,
    inventor_id TEXT,
    company_id  TEXT
);
