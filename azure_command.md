# Azure commands :

# Commands :

 1977  az extension add --name azure-devops
 1978  conda deactivate
 1979  az devops --help
 1980  az devops login
 1981  az devops configure --defaults   organization=https://dev.azure.com/sarathkumar-test   project=databricks-cicd
 1982  az devops configure --list
 1983  az devops user show
 1984  az devops login
 1985  az devops user show
 1986  az repos list --output table
 1987  az repos delete   --repository Databricks-CICD   --yes
 1988  az repos delete   --id 8c9538c4-6ff2-49f9-84b8-6e2bb6fad4c9   --yes
 1989  ORG="sarathkumar-test"
 1990  PROJECT="Databricks-CICD"
 1991  REPO_ID="8c9538c4-6ff2-49f9-84b8-6e2bb6fad4c9"
 1992  PAT="YOUR_PAT_HERE"
 1993  curl -u :$PAT   -X DELETE   "https://dev.azure.com/$ORG/$PROJECT/_apis/git/repositories/$REPO_ID?api-version=7.1-preview.1"
 1994  az repos list --output table
 1995  curl -s -u :$AZDO_PAT   -X DELETE   "https://dev.azure.com/sarathkumar-test/Databricks-CICD/_apis/git/repositories/8c9538c4-6ff2-49f9-84b8-6e2bb6fad4c9?api-version=7.0"
 1996  history


 ## Managed identity page for **Microsoft Entra Id** permission 
 #  **https://learn.microsoft.com/en-us/azure/devops/pipelines/release/configure-workload-identity?view=azure-devops&tabs=managed-identity**   

## Azure create resouce group by cli 
az group create \
  --name <resource group name> \
  --location eastus


 ## Azure VM machine creation

az vm create \
  --resource-group <resource group name> \
  --name ado-agent-vm \
  --image Ubuntu2204 \
  --size Standard_D2s_v5 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --location eastus

## if above through error go and create virtual machine **QUOTE** to assing we need --> Subscription -> settings -> Usage + Quote -> select comput -> click (New Quote Request in top right) and give time you want. 


## verify VM is available or not 
- az vm list-sizes --location australiacentral --output table

## create VM 
az vm create \
  --resource-group databricks-cicd-test \
  --name ado-agent-vm \
  --image Ubuntu2204 \
  --size Standard_B2s_v2 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --location australiacentral

## Vritual machien details 

sarathkumar-r@Sarathkumar:~/Public/Honeywell_ML/Azure_databricks/Databricks-CICD-test$ az vm create   --resource-group databricks-cicd-test   --name ado-agent-vm   --image Ubuntu2204   --size Standard_B2s_v2   --admin-username azureuser   --generate-ssh-keys   --location australiacentral
The default value of '--size' will be changed to 'Standard_D2s_v5' from 'Standard_DS1_v2' in a future release.
{
  "fqdns": "",
  "id": "/subscriptions/4a4c77c9-cb05-45eb-80b6-5c22fc9061c2/resourceGroups/databricks-cicd-test/providers/Microsoft.Compute/virtualMachines/ado-agent-vm",
  "location": "australiacentral",
  "macAddress": "7C-ED-8D-61-D9-6B",
  "powerState": "VM running",
  "privateIpAddress": "10.0.0.4",
  "publicIpAddress": "20.28.59.28",
  "resourceGroup": "databricks-cicd-test"
}

## delete virtual machine 
az vm delete \
  --resource-group databricks-cicd-test \
  --name ado-agent-vm \
  --yes

az network nic delete \
  --resource-group databricks-cicd-test \
  --name ado-agent-vmVMNic

az network public-ip delete \
  --resource-group databricks-cicd-test \
  --name ado-agent-vm-ip
