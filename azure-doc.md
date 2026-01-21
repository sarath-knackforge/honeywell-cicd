# Azure Databricks Setup Guide

## 1. Create Azure Account
- Create a **Pay-As-You-Go Azure account**.

## 2. Verify Microsoft Entra ID Permissions
- Open **Microsoft Entra ID**.
- Check if you have **Global Administrator** permission.
- If not:
  - Raise an **access request**.
  - Without this permission, you **cannot access Databricks Account Management**.

## 3. Create Resource Group
- Create a new **Resource Group** in Azure.

## 4. Create Azure Databricks Workspace
- Create an **Azure Databricks** workspace.
- Select the **Resource Group** where the workspace should be created.

## 5. Create Storage Account (For Unity Catalog)
- Create a **Storage Account**.
- Select the required **Resource Group**.
- Provide a **Storage Account name**.
- Choose:
  - **Standard** or **Premium** *(Premium recommended for low latency)*.
- Select **LRS (Locally Redundant Storage)**.
- Click **Next**.
- Enable **Hierarchical Namespace** (**Mandatory for Unity Catalog**).

## 6. Create Container for Unity Metastore
- After deployment, click **Go to resource**.
- Navigate to **Data Storage**.
- Create a **Container** for the **Unity Metastore**.

## 7. Create Databricks Access Connector
- Create a **Databricks Access Connector**.
- This connector is used to **access Azure storage and other resources**.

## 8. Grant Storage Access to Databricks Access Connector
- Go to the created **Storage Account**.
- Open **IAM (Access Control)**.
- Click **+ Add** → **Add role assignment**.
- Search and select **Storage Blob Data Contributor**.
- Go to **Members**:
  - Select **Managed Identity**.
  - Click **+ Select members**.
  - Choose **Access Connector for Azure Databricks**.
  - Select the **Databricks Access Connector** you created.
- Complete the **role assignment**.

## 9. Launch Azure Databricks
- Search for **Azure Databricks** in the Azure portal.
- Click **Launch** to open the workspace.

## 10. Install Azure Cli using Curl:
- curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash