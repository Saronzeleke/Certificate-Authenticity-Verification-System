-- Create providers table
CREATE TABLE IF NOT EXISTS providers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Now add the analysis fields from your migration
ALTER TABLE providers 
ADD COLUMN IF NOT EXISTS credential_analysis JSONB,
ADD COLUMN IF NOT EXISTS analysis_status VARCHAR(50),
ADD COLUMN IF NOT EXISTS analysis_timestamp TIMESTAMP,
ADD COLUMN IF NOT EXISTS analysis_notes TEXT;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_providers_analysis_status ON providers(analysis_status);
CREATE INDEX IF NOT EXISTS idx_providers_analysis_timestamp ON providers(analysis_timestamp);
