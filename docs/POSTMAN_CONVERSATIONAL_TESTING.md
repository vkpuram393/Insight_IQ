# Postman Conversational Testing Guide

## Quick Setup for Testing with Same Session

This guide shows you how to test conversational capabilities where the agent maintains context across multiple prompts.

---

## 🎯 What You'll Achieve

- All requests use the **same session ID**
- Agent maintains **conversation context** across requests
- Test **follow-up questions**, **clarifications**, and **multi-turn conversations**

---

## ✅ Step-by-Step Setup

### Step 1: Create Your Conversational Prompts CSV

Create a CSV file with prompts that reference previous context:

```csv
prompt
What is my claim status?
Tell me more about it
What was the rejection reason?
Can you explain that in simpler terms?
```

Save as `conversation_test.csv`.

**Note**: Use prompts that reference previous context (like "it", "that", "more about it") to test conversational capabilities.

### Step 2: Set Up Postman Collection Variable

1. **Right-click your collection** → **Edit**
2. Go to **Variables** tab
3. Click **Add**
4. Set:
   - **Variable**: `shared_session_id`
   - **Initial Value**: (leave empty)
   - **Current Value**: (leave empty)
5. Click **Save**

### Step 3: Create/Update Your Batch Test Request

1. **Method**: POST
2. **URL**: `http://localhost:8000/api/v1/test/batch`
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
   - Enter this JSON:
   ```json
   {
     "text": "{{prompt}}",
     "session_id": "{{shared_session_id}}"
   }
   ```
   
   **CRITICAL - Must Match Exactly:**
   - ✅ Field name must be `"text"` (not `"prompt"`)
   - ✅ Value must be `"{{prompt}}"` with double curly braces (no spaces)
   - ✅ Add `"session_id": "{{shared_session_id}}"` for conversational testing

5. **Save** the request

### Step 4: Add Pre-request Script

1. In your request, go to **Pre-request Script** tab
2. Add this code:
```javascript
// Generate session ID on first request, reuse for all subsequent requests
if (!pm.collectionVariables.get("shared_session_id")) {
    // Generate a new UUID for the session
    const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
    pm.collectionVariables.set("shared_session_id", uuid);
    console.log("Generated new session ID:", uuid);
} else {
    console.log("Reusing existing session ID:", pm.collectionVariables.get("shared_session_id"));
}
```

3. **Save** the request

### Step 5: Run Collection Runner

1. **Open Collection Runner**:
   - Click your collection name (left sidebar)
   - Click **"Run"** button (top right)

2. **Select Your Request**:
   - Make sure your batch test request is **checked** ✅

3. **Select Data File**:
   - Click **"Select File"** button
   - Choose your `conversation_test.csv` file
   - **VERIFY**: Postman should show: **"1 variable detected: prompt"**
   - **You should see a preview table** showing your prompts listed

4. **Set Delay**:
   - **Delay**: Set to `1000` milliseconds (1 second between requests)
   - This prevents overwhelming the server

5. **Click Run Batch Testing**

**Result**: All requests will use the same `shared_session_id`, maintaining conversation context!

---

## 🔍 Verify It's Working

### Check 1: First Request Response
Look at the first response - it should have a `session_id`:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_prompt": "What is my claim status?",
  "agent_response": "I can help you check your claim status...",
  ...
}
```

### Check 2: All Requests Use Same Session ID
All subsequent requests should have the **same** `session_id` value. Export results and verify all `session_id` values are identical.

### Check 3: Conversational Context Works
Test with prompts that reference previous context:
- **Request 1**: `"What is my claim status?"` → Agent responds with claim status
- **Request 2**: `"Tell me more about it"` → Agent should understand "it" refers to the claim from Request 1
- **Request 3**: `"What was the rejection reason?"` → Agent should understand this refers to the claim context
- **Request 4**: `"Can you explain that in simpler terms?"` → Agent should explain the rejection reason in simpler terms

### Check 4: View in Postman Console
1. **View** → **Show Postman Console**
2. Look at the **Pre-request Script** logs:
   - First request: "Generated new session ID: ..."
   - Subsequent requests: "Reusing existing session ID: ..."
3. Look at the actual request being sent:
   - Should see `"session_id": "same-uuid-here"` in all requests

---

## 🎨 Alternative: Manual Session ID (Simpler Method)

If you prefer to set a fixed session ID manually (no pre-request script needed):

1. **Generate a UUID**:
   - Use any UUID generator online
   - Or use `{{$randomUUID}}` in Postman
   - Or use a simple ID like `azhagu-1125129`

2. **Set Collection Variable**:
   - Right-click collection → **Edit**
   - Go to **Variables** tab
   - Set `shared_session_id` = `your-uuid-here` (e.g., `azhagu-1125129`)
   - Click **Save**

3. **Use in Request Body** (same as above):
   ```json
   {
     "text": "{{prompt}}",
     "session_id": "{{shared_session_id}}"
   }
   ```

4. **No Pre-request Script Needed**: The session ID is already set, so you can skip Step 4.

**Advantages**:
- ✅ Simpler setup (no JavaScript code)
- ✅ You control the exact session ID
- ✅ Easier to debug (you know the session ID upfront)

**Disadvantages**:
- ❌ Must manually set the session ID
- ❌ Can't easily switch between different session IDs

---

## 📊 Example: Testing Conversation Flow

### CSV File (`conversation_test.csv`):
```csv
prompt
What is my claim status?
Tell me more about it
What was the rejection reason?
Can you explain that in simpler terms?
```

### Expected Behavior:
- **Request 1**: Agent responds to "What is my claim status?"
- **Request 2**: Agent understands "it" refers to the claim from Request 1
- **Request 3**: Agent understands "that" refers to the rejection reason
- **Request 4**: Agent provides simpler explanation based on previous context

All using the **same session ID**!

---

## 🔄 Switching Back to Independent Testing

To test each prompt independently (new session each time):

1. **Remove `session_id` from request body**:
   ```json
   {
     "text": "{{prompt}}"
   }
   ```

2. **Remove or comment out Pre-request Script**

3. Each request will now get a new session ID

---

## 🆘 Troubleshooting

### Problem: Still getting different session IDs

**Solution**: 
- ✅ Check Pre-request Script is saved (if using auto-generation method)
- ✅ Verify collection variable name is exactly `shared_session_id` (case-sensitive)
- ✅ Check request body uses `{{shared_session_id}}` (with curly braces, no spaces)
- ✅ Verify collection variable exists and is set (check Variables tab)
- ✅ Try manual session ID method instead (simpler, no script needed)

### Problem: Session ID not being set

**Solution**:
- ✅ Check Pre-request Script syntax (no JavaScript errors in console)
- ✅ Verify collection variable exists in Variables tab
- ✅ Check Postman Console for errors (View → Show Postman Console)
- ✅ Try manual session ID method instead (set it directly in Variables tab)

### Problem: Agent not maintaining context

**Solution**:
- ✅ Verify all requests show same `session_id` in responses (export results and check)
- ✅ Check that memory store is configured (inmemory, redis, etc. in `config.py`)
- ✅ Ensure session ID is being passed correctly (check request body in Postman Console)
- ✅ Test with simple prompts first: "What is my claim status?" then "Tell me more about it"

### Problem: Variables not being substituted

**Solution**:
- ✅ Must use **Collection Runner** (variables don't work in manual requests)
- ✅ CSV file must have header exactly `prompt` (lowercase)
- ✅ Request body must use `{{prompt}}` and `{{shared_session_id}}` (with curly braces)
- ✅ Check Collection Runner shows "1 variable detected: prompt"

---

## 💡 Pro Tips

1. **Test with Small CSV First**: 
   - Use 5-10 prompts to verify it works before running all prompts
   - Create a test CSV with just 2-3 conversational prompts
   - Verify same session ID appears in all responses

2. **Check Session ID in Responses**: 
   - Export results to CSV
   - Verify all `session_id` values are identical
   - If they differ, check your setup

3. **Mix Independent and Conversational**: 
   - You can have some requests with session, some without
   - Just remove `"session_id": "{{shared_session_id}}"` from body for independent requests

4. **Use Different Session IDs**: 
   - Create multiple collection variables for different conversation threads
   - Example: `shared_session_id_thread1`, `shared_session_id_thread2`
   - Switch between them by changing the variable name in request body

5. **Monitor Conversation Flow**:
   - Watch Postman Console to see session ID being reused
   - Check agent responses to verify context is maintained
   - Test with follow-up questions like "Tell me more" or "What about that?"

---

## 📝 Quick Reference

**Independent Testing** (New Session Each Time):
```json
{
  "text": "{{prompt}}"
}
```

**Conversational Testing** (Same Session):
```json
{
  "text": "{{prompt}}",
  "session_id": "{{shared_session_id}}"
}
```

**Pre-request Script** (Auto-generate session ID):
```javascript
if (!pm.collectionVariables.get("shared_session_id")) {
    const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
    pm.collectionVariables.set("shared_session_id", uuid);
}
```

---

That's it! You're ready to test conversational capabilities! 🚀

