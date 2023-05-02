# Setting up your Fork

https://docs.github.com/en/get-started/quickstart/fork-a-repo#configuring-git-to-sync-your-fork-with-the-upstream-repository

1. Create a fork on github through the UI

    Creates a fork that you now own and can modify, add collaborators etc.

2. `git clone <fork_address>`

    Clones the repository like any other github repo that you own

3. `git branch`

    You have a main and can create branches of your own to work off of and merge to main.

4. `git remote -v`

    You can see that origin is pinned to your fork. However, as of right now, your fork can't be updated when updates to the main repository occur. So we need to set an "upstream"

5. `git remote add upstream git@github.com:VectorInstitute/PromptEngineering.git`

    This adds the forked repository as an upstream to our main branch so that we can pull in changes from the original repo when we want to "sync" the fork

6. `git remote -v`

    Now you can see the original repository as an upstream to our fork.

# Working in your Fork, just like any old repository

7. `git checkout -b <branch_name>`

8. Make some changes and commit

9. `git push -u origin <branch_name>`

# Steps to sync a Fork

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

Say that there are changes in the main branch of the original repository that you would like to bring in.

1. `git fetch upstream`

    In your fork, we bring in the changes to our fork. They are stored locally as branches named `upstream/main`

2. `git checkout main`

    Checkout the main branch to bring in the changes from upstream

3. `git merge upstream/main`

    Merge the new changes into main as you would with a normal merge

# Proposing changes to the original repository

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork

Pull requests function quite similarly to a standard pull request on a repository you own. You just have to specify the base repository as well as the target and source branches.
