# Postman Batch Testing Troubleshooting

## Problem: All Requests Return Same Response with `{{prompt}}` Literal

If you see `"user_prompt": "{{prompt}}"` in all responses, Postman is not substituting the CSV variable.

## Quick Fix Checklist

### ✅ Step 1: Verify CSV File Format

Your CSV file must have:
- **First row is header**: `prompt` (exactly this name, lowercase)
- **One prompt per row** (no quotes needed)
- **File saved as `.csv`** (not `.txt`)

**Example `prompts.csv`:**
```csv
prompt
What is my claim status?
Tell me about claim 12345
How much did I pay?
```

### ✅ Step 2: Verify Request Body

In Postman request body (Body → raw → JSON):

**CORRECT:**
```json
{
  "text": "{{prompt}}"
}
```

**WRONG:**
```json
{
  "text": "prompt"
}
```
or
```json
{
  "prompt": "{{prompt}}"
}
```

The field name must be `text` (not `prompt`), and the value must be `{{prompt}}` with double curly braces.

### ✅ Step 3: Verify Collection Runner Setup

1. **Open Collection Runner**:
   - Click your collection name (left sidebar)
   - Click **"Run"** button (top right)

2. **Select Your Request**:
   - Make sure your batch test request is **checked** ✅

3. **Select Data File**:
   - Click **"Select File"** button
   - Choose your `prompts.csv` file
   - Postman should show: **"1 variable detected: prompt"**
   - If it says "No variables detected", your CSV header is wrong

4. **Verify Variable Mapping**:
   - You should see a preview table showing:
     - Column: `prompt`
     - First few rows of your prompts
   - If you see "No data" or empty rows, your CSV format is wrong

### ✅ Step 4: Test Variable Substitution

Before running all 596 requests, test with 1-2 rows:

1. **Temporarily edit your CSV** to have only 2 rows:
   ```csv
   prompt
   Test prompt 1
   Test prompt 2
   ```

2. **Run Collection Runner** with this small CSV

3. **Check the response**:
   - Should see `"user_prompt": "Test prompt 1"` (not `{{prompt}}`)
   - If you still see `{{prompt}}`, go back to Steps 1-3

### ✅ Step 5: Common Mistakes

**Mistake 1: CSV Header Wrong**
```csv
Prompt  ← Wrong (capital P)
prompts ← Wrong (plural)
text    ← Wrong (should be "prompt")
```

**Mistake 2: Request Body Field Wrong**
```json
{
  "prompt": "{{prompt}}"  ← Wrong field name
}
```

**Mistake 3: Not Using Collection Runner**
- Running request manually won't substitute variables
- Must use Collection Runner

**Mistake 4: CSV File Not Selected**
- Collection Runner must have CSV file selected
- Check "Data File" section shows your file

## Verification Steps

After fixing, verify:

1. **In Collection Runner preview**, you should see your prompts listed
2. **First request response** should have `"user_prompt"` with actual prompt text (not `{{prompt}}`)
3. **Different requests** should have different `user_prompt` values

## Still Not Working?

1. **Check Postman Console**:
   - View → Show Postman Console
   - Look at the actual request being sent
   - Should see `"text": "actual prompt here"` (not `{{prompt}}`)

2. **Try Pre-request Script** (alternative method):
   ```javascript
   // In Pre-request Script tab
   pm.variables.set("prompt", pm.iterationData.get("prompt"));
   ```
   Then in body use: `{{prompt}}`

3. **Export and Check CSV**:
   - Open CSV in Excel/Text Editor
   - Verify first row is exactly: `prompt`
   - Verify no extra spaces or quotes

## Expected Behavior

**Before Fix:**
```json
{
  "user_prompt": "{{prompt}}",
  "agent_response": "Hello! I am ready to assist..."
}
```

**After Fix:**
```json
{
  "user_prompt": "What is my claim status?",
  "agent_response": "I can help you check your claim status..."
}
```

Each request should have a different `user_prompt` value matching your CSV row.

