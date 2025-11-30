# Postman Batch Test Setup Guide

This guide walks you through setting up Postman to test 700 prompts using the batch test endpoint.

## 📋 What You'll Need

1. **Postman** installed on your computer (free version works)
2. **700 prompts** in a file (CSV or JSON)
3. **Your API URL** (e.g., `http://localhost:8000`)

---

## Step 1: Prepare Your Prompts File

### Option A: CSV File (Recommended - Easy for Excel)

Create a file called `prompts.csv` with this format:

```csv
prompt
What is my claim status?
Tell me about claim 12345
How much did I pay for my prescription?
What are my benefits?
Can I refill my medication?
```

**CRITICAL - Must Follow Exactly:**
- ✅ First row must be the header: `prompt` (lowercase, exactly this word)
- ✅ One prompt per row (no quotes needed unless prompt contains commas)
- ✅ Save as `.csv` file (not `.txt`)
- ❌ **DO NOT** use `Prompt`, `prompts`, `text`, or any other header name
- ❌ **DO NOT** add extra spaces or quotes around the header

**If you see `{{prompt}}` in responses, your CSV header is wrong!**

### Option B: JSON File

Create a file called `prompts.json`:

```json
[
  {"prompt": "What is my claim status?"},
  {"prompt": "Tell me about claim 12345"},
  {"prompt": "How much did I pay for my prescription?"}
]
```

---

## Step 2: Create the Request in Postman

### 2.1 Open Postman

1. Open Postman application
2. Click **"New"** button (top left)
3. Select **"HTTP Request"**

### 2.2 Configure the Request

1. **Method**: Select **POST** from dropdown
2. **URL**: Enter your API URL + endpoint:
   ```
   http://localhost:8000/api/v1/test/batch
   ```
   (Replace `localhost:8000` with your actual server URL)

3. **Headers Tab**:
   - Click **"Headers"** tab
   - Add header:
     - **Key**: `Content-Type`
     - **Value**: `application/json`

4. **Body Tab**:
   - Click **"Body"** tab
   - Select **"raw"** radio button
   - Select **"JSON"** from dropdown (right side)
   - Enter this JSON (copy exactly):
   ```json
   {
     "text": "{{prompt}}"
   }
   ```
   
   **CRITICAL - Must Match Exactly:**
   - ✅ Field name must be `"text"` (not `"prompt"`)
   - ✅ Value must be `"{{prompt}}"` with double curly braces (no spaces)
   - ✅ The `{{prompt}}` variable name must match your CSV header column name
   
   **If you see `{{prompt}}` in responses, check:**
   1. CSV header is exactly `prompt` (lowercase)
   2. Request body field is `text` (not `prompt`)
   3. Request body value is `{{prompt}}` (with curly braces)
   4. CSV file is selected in Collection Runner

### 2.3 Save the Request

1. Click **"Save"** button
2. Name it: `Batch Test - Single Prompt`
3. Save to a collection (create new collection if needed: "Batch Testing")

---

## Step 3: Set Up Collection Runner

### 3.1 Open Collection Runner

1. Click on your collection name (left sidebar)
2. Click **"Run"** button (top right)
3. This opens the **Collection Runner** window

### 3.2 Configure the Runner

1. **Select Request**: Make sure your "Batch Test - Single Prompt" request is checked

2. **Data File**:
   - Click **"Select File"** button
   - Choose your `prompts.csv` file
   - **VERIFY**: Postman should show: **"1 variable detected: prompt"**
   - **If it says "No variables detected"**: Your CSV header is wrong - check it's exactly `prompt` (lowercase)
   - **You should see a preview table** showing your prompts listed

3. **Iterations**:
   - **Iterations**: Leave as `1` (runs once per row)
   - **Delay**: Set to `1000` milliseconds (1 second between requests)
     - This prevents overwhelming the server
     - You can adjust based on your server capacity

4. **Data File Type**: Should auto-detect as CSV

### 3.3 Run the Collection

1. Click **"Run Batch Testing"** button (bottom right)
2. Postman will:
   - Send request #1 with first prompt
   - Wait 1 second
   - Send request #2 with second prompt
   - Continue for all 700 prompts

---

## Step 4: View Results

### 4.1 During Execution

- You'll see each request being sent
- Green checkmark = success
- Red X = error
- You can see response time for each request

### 4.2 After Completion

1. Click **"Export Results"** button (top right of Collection Runner)
2. Choose **"Export as CSV"**
3. Save the file (e.g., `batch_test_results.csv`)

---

## Step 5: Open Results in Excel

1. Open Excel
2. File → Open → Select `batch_test_results.csv`
3. Excel will import the CSV
4. You'll see columns like:
   - `prompt` (your original prompt)
   - `response` (agent response)
   - `embedding_classifier_confidence`
   - `llm_judge_confidence`
   - `intent`
   - `clarification_called`
   - `exception` (if any error occurred)

---

## 📊 Understanding the Response

Each response will look like this:

```json
{
  "user_prompt": "What is my claim status?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_response": "I can help you check your claim status...",
  "embedding_classifier_confidence": 0.85,
  "llm_judge_confidence": 0.92,
  "intent": "claim_status",
  "clarification_called": false,
  "exception": null
}
```

**Note**: The `session_id` is returned in every response. Use it for conversational testing (see below).

### If an Error Occurs:

```json
{
  "user_prompt": "What is my claim status?",
  "agent_response": null,
  "embedding_classifier_confidence": null,
  "llm_judge_confidence": null,
  "intent": null,
  "clarification_called": false,
  "exception": {
    "error_type": "HTTPException",
    "message": "Error message here",
    "stacktrace": "..."
  }
}
```

---

## 💬 Conversational Testing (Same Session)

To test conversational capabilities where the agent maintains context across multiple prompts:

### Option 1: Use Postman Collection Variable (Recommended)

1. **Create a Collection Variable**:
   - Right-click your collection → **Edit**
   - Go to **Variables** tab
   - Add variable:
     - **Name**: `shared_session_id`
     - **Value**: (leave empty - will be set automatically)
     - **Type**: String

2. **Update Request Body**:
   ```json
   {
     "text": "{{prompt}}",
     "session_id": "{{shared_session_id}}"
   }
   ```

3. **Add Pre-request Script** (in your request):
   ```javascript
   // Generate session ID on first request, reuse for subsequent requests
   if (!pm.collectionVariables.get("shared_session_id")) {
       pm.collectionVariables.set("shared_session_id", pm.variables.replaceIn("{{$randomUUID}}"));
   }
   ```

4. **Run Collection Runner**:
   - All requests will use the same session ID
   - Agent will maintain conversation context
   - You can test follow-up questions, clarifications, etc.

### Option 2: Manual Session ID

1. **Generate a Session ID** (use any UUID generator or `{{$randomUUID}}` in Postman)

2. **Update Request Body**:
   ```json
   {
     "text": "{{prompt}}",
     "session_id": "your-fixed-session-id-here"
   }
   ```

3. **Run Collection Runner**:
   - All requests will use the same session ID
   - Agent maintains context across all prompts

### Option 3: Extract and Reuse Session ID

1. **First Request**: Don't provide `session_id` (gets new one)
2. **Extract Session ID** from first response
3. **Subsequent Requests**: Use extracted session ID

**Example Postman Test Script** (to extract and save session ID):
```javascript
// In Tests tab of your request
if (pm.response.code === 200) {
    const response = pm.response.json();
    if (response.session_id) {
        // Save to collection variable for reuse
        pm.collectionVariables.set("shared_session_id", response.session_id);
    }
}
```

Then in request body:
```json
{
  "text": "{{prompt}}",
  "session_id": "{{shared_session_id}}"
}
```

### When to Use Each Approach

- **Independent Testing**: Don't provide `session_id` (each request gets new session)
- **Conversational Testing**: Use same `session_id` for all requests (maintains context)
- **Mixed Testing**: Some requests with session, some without

---

## 🎯 Tips & Troubleshooting

### Tip 1: Start Small
- Test with 5-10 prompts first
- Make sure everything works
- Then run all 700

### Tip 2: Adjust Delay
- If server is slow: Increase delay to 2000ms (2 seconds)
- If server is fast: Decrease delay to 500ms (0.5 seconds)

### Tip 3: Handle Errors
- If a request fails, Postman will show it in red
- Check the response to see what went wrong
- Failed requests won't stop the batch - it continues

### Tip 4: Save Progress
- Postman saves results automatically
- You can stop and resume later
- Export results periodically to avoid losing data

### Tip 5: Monitor Server
- Watch your server logs during batch testing
- Make sure server can handle the load
- Consider running during off-peak hours

---

## 🔍 What Gets Logged?

**Nothing!** The batch test endpoint:
- ✅ Processes requests through full agentic flow
- ✅ Returns all response data
- ❌ Does NOT log to `logs` table
- ❌ Does NOT log to `exceptions` table
- ❌ Does NOT create telemetry events

This keeps your database clean during testing.

---

## 📝 Quick Reference

**Endpoint**: `POST /api/v1/test/batch`

**Request Body**:
```json
{
  "text": "your prompt here"
}
```

**Request Body** (Independent Testing - New Session Each Time):
```json
{
  "text": "{{prompt}}"
}
```

**Request Body** (Conversational Testing - Same Session):
```json
{
  "text": "{{prompt}}",
  "session_id": "{{shared_session_id}}"
}
```

**Response**:
```json
{
  "user_prompt": "...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_response": "...",
  "embedding_classifier_confidence": 0.85,
  "llm_judge_confidence": 0.92,
  "intent": "claim_status",
  "clarification_called": false,
  "exception": null
}
```

---

## ✅ Checklist

Before running 700 prompts:

- [ ] Postman installed
- [ ] Prompts file created (CSV or JSON)
- [ ] Request created in Postman
- [ ] URL is correct
- [ ] Headers set (Content-Type: application/json)
- [ ] Body uses `{{prompt}}` variable
- [ ] Tested with 1-2 prompts first
- [ ] Delay set appropriately
- [ ] Server is running and accessible
- [ ] Ready to export results to CSV

---

## 🆘 Need Help?

### Problem: All responses show `{{prompt}}` instead of actual prompts

**This means Postman isn't substituting the CSV variable. Fix:**

1. ✅ **Check CSV header**: Must be exactly `prompt` (lowercase, no spaces)
2. ✅ **Check request body**: Field must be `"text"`, value must be `"{{prompt}}"`
3. ✅ **Check Collection Runner**: CSV file must be selected in "Data File"
4. ✅ **Check variable detection**: Should show "1 variable detected: prompt"
5. ✅ **Test with 2 rows first**: Create small CSV with 2 prompts to verify it works

See `docs/POSTMAN_TROUBLESHOOTING.md` for detailed troubleshooting.

### Other Issues:

1. **Check the URL**: Make sure it's correct (`/api/v1/test/batch`)
2. **Check the server**: Is it running?
3. **Test manually**: Try one request in Postman first (without Collection Runner) - but note: variables won't work without Collection Runner
4. **Check CSV format**: Make sure first row is `prompt` (exactly, lowercase)
5. **Check variable name**: Must be `{{prompt}}` in the body (with curly braces)

---

## 🎉 You're Ready!

Once you've completed the checklist, you're ready to run your 700 prompts. The whole process will take about 10-15 minutes (depending on delay settings and server speed).

Good luck with your testing! 🚀

