#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

// Get migration name from command line arguments
const migrationName = process.argv[2];

if (!migrationName) {
    console.error('Error: Migration name is required');
    console.error('Usage: bun run migrate:make <migration_name>');
    console.error('Example: bun run migrate:make create_users_table');
    process.exit(1);
}

// Generate timestamp in format YYYYMMDDHHMMSS
const generateTimestamp = () => {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');

    return `${year}${month}${day}${hours}${minutes}${seconds}`;
};

// Template for TypeScript migration using system line endings
const migrationTemplate = [
].join(os.EOL);

try {
    // Change to the api directory to ensure correct working directory
    const apiDir = path.dirname(__dirname);
    process.chdir(apiDir);

    // Generate filename with timestamp
    const timestamp = generateTimestamp();
    const fileName = `${timestamp}_${migrationName}.ts`;
    const migrationsDir = path.join(apiDir, 'src', 'migrations');
    const filePath = path.join(migrationsDir, fileName);

    // Ensure migrations directory exists
    if (!fs.existsSync(migrationsDir)) {
        fs.mkdirSync(migrationsDir, { recursive: true });
    }

    // Create the migration file
    fs.writeFileSync(filePath, migrationTemplate);
    console.log(`Migration created: ${fileName}`);

    // Add to git
    try {
        execSync(`git add ${filePath}`, { encoding: 'utf8' });
        console.log('Migration added to git VCS');
    } catch (gitError) {
        console.log(`Warning: Could not add to git (${gitError.message})`);
    }

} catch (error) {
    console.error('Failed to create migration:', error.message);
    process.exit(1);
}
