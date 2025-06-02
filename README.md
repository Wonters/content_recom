# content_recom
Project base on content recommendation 


# Create an azure fonction 
Assume you account on azure cloud is valid
Presume contentrecom as storage account name and poc and resource-group 
Assume azure-cli and and azure functions core tools is installed.
Create a functionapp 
```bash
az login
az functionapp create --name contentrecomapp-new --storage-account contentrecom --resource-group poc --consumption-plan-location westeurope --runtime python --runtime-version 3.10.15 --functions-version 4 --os-type linux
```

## Update file on blob 
```bash
az storage blob upload --account-name poc  --container-name uploads --name index.hnsw --file ../index.hnsw
```

## Deploy an azure function 
```bash
func azure functionapp publish contentrecomapp-new --build remote
```