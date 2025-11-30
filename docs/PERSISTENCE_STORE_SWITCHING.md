# Switching Between SQLite and MongoDB

This guide explains how to switch between SQLite and MongoDB persistence stores.

## Quick Switch

### Use SQLite (Default)

In your `.env` file:
```bash
PERSISTENCE_STORE_TYPE=sqlite
```

### Use MongoDB

In your `.env` file:
```bash
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_CONNECTION_STRING=mongodb+srv://myclaims_dev:<PASSWORD>@mdb-use4-myclaims-dev01-pl-0.knpouh.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE_NAME=myclaims-DEV
```

## Persistence Store Types

| Type | Description | Use Case |
|------|-------------|----------|
| `sqlite` | Local SQLite database file | Development, testing, no server required |
| `mongodb` | MongoDB database server | Production, requires MongoDB server and connection |
| `firestore` | Google Cloud Firestore | Future implementation |
| `bigquery` | Google BigQuery | Future implementation |

## Configuration Options

| Setting | Default | Purpose |
|---------|---------|---------|
| `PERSISTENCE_STORE_TYPE` | `sqlite` | Which persistence store to use |
| `MONGODB_CONNECTION_STRING` | `mongodb://localhost:27017` | MongoDB connection string (only for mongodb type) |
| `MONGODB_DATABASE_NAME` | `myclaims-DEV` | MongoDB database name (only for mongodb type) |

## Workflow

### Step 1: Test MongoDB Connectivity

Use a GUI client (DBeaver, MongoDB Compass, VS Code extension) or test script to verify:
- Connection works
- Permissions are correct
- Can insert/read documents

### Step 2: Switch to MongoDB

Update `.env`:
```bash
PERSISTENCE_STORE_TYPE=mongodb
MONGODB_CONNECTION_STRING=...
MONGODB_DATABASE_NAME=...
```

### Step 3: Test Application

```bash
# Run test script
python scripts/test_mongodb_connection.py

# If successful, restart your application
```

## Switching Back to SQLite

Simply update `.env`:
```bash
PERSISTENCE_STORE_TYPE=sqlite
```

No other changes needed - the application will automatically use SQLite.

## Error Messages

### If Connection Fails

```
❌ Failed to connect to MongoDB: ...
```

**Solution**: 
1. Verify connection string is correct
2. Check username and password
3. Ensure network connectivity
4. Verify permissions

## Best Practices

1. **Start with SQLite**: Use SQLite by default for development
2. **Test MongoDB separately**: Use test scripts and GUI clients to verify connectivity before switching
3. **Switch when ready**: Only change `PERSISTENCE_STORE_TYPE=mongodb` when connectivity is verified

