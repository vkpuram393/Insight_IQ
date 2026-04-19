from VamsiSir import embeddingVars
import pandas as pd

DomainsVamsi = [v for v in embeddingVars.DOMAIN_REGISTRY]
IntentsVamsi =set() 

for domain in embeddingVars.DOMAIN_REGISTRY:
    for intent in embeddingVars.DOMAIN_REGISTRY[domain]["intents"]:
        IntentsVamsi.add(intent)

df = pd.read_excel("600_prompts_responses.xlsx", sheet_name="600 Prompts Responses")

df[df["Intent"].isin(IntentsVamsi)][["Prompt","Intent"]].to_csv("Testdata.csv", index=False)