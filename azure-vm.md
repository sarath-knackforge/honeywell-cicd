az login
az devops login
az ad app list

# list of VM
az vm list-usage \
  --location australiacentral \
  --output table

  # Create VM

  az vm create \
  --resource-group databricks-cicd-test \
  --name ado-agent-vm \
  --image Ubuntu2204 \
  --size Standard_B2s_v2 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --location australiacentral

  # delte VM

  az network nic delete \
  --resource-group databricks-cicd-test \
  --name ado-agent-vmVMNic

az network public-ip delete \
  --resource-group databricks-cicd-test \
  --name ado-agent-vm-ip


# connect VM

ssh azureuser@<public ip addres of VM>

# EXIT VM 
 Ctrl + D