# Git commands

# 1. git branch
# 2. Create branch ** git checkout -b <branch-name>**
# 3. git checkout <branch-name>
# 4. git push origin <branch-name>

# 5. Raise a PR to main branch with different aspects:

    - Basic PR (most common)
    az repos pr create \
  --source-branch <feature-branch-name> \
  --target-branch main \
  --title "Add Azure Databricks setup documentation" \
  --description "Added step-by-step Azure Databricks and Unity Catalog setup guide in markdown format."

# 6. Create PR with reviewers (recommended)

    az repos pr create \
  --source-branch <feature-branch-name> \
  --target-branch main \
  --title "Add Azure Databricks setup documentation" \
  --description "This PR adds Azure Databricks + Unity Catalog setup steps." \
  --reviewers user1@company.com user2@company.com

# 7. Create PR and auto-complete after approval (Enterprise)

    az repos pr create \
  --source-branch <feature-branch-name> \
  --target-branch main \
  --title "Add Azure Databricks setup documentation" \
  --description "Documentation for Azure Databricks setup" \
  --reviewers lead@company.com \
  --auto-complete true \
  --delete-source-branch true

# 8. PR merge approval uisng cli
#   az repos pr set-vote --id <PR ID> --vote approve 
                    (OR) 
#   az repos pr reviewer add --id 2 --reviewer sarathkumar.r@knackforge.com


# 9. Verify PR from CLI
#   To see all reviewer ==> az repos pr show --id <PR_ID> --query "reviewers[].{Reviewer:displayName, Vote:vote}" --output table
#   To check atleast one approval ==> az repos pr show --id <PR_ID> --query "reviewers[?vote==\`10\`].displayName"

#   ** az repos pr list --output table **
#   ** az repos pr show --id <PR_ID> **

# 10. PR approval status
#   az repos pr show --id <PD ID> --query status
- the above command will show result **completed**
