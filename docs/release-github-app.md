# Release automation GitHub App prototype

## Review and plan

Headroom runs `googleapis/release-please-action@v5` on pushes to `main`.
It maintains the release PR and publishes the GitHub release when that PR
merges. `release.yml` and `docker.yml` consume the published release event.
`release-metadata-sync.yml` pushes generated version files to release branches.
Both writers currently attempt App authentication but silently fall back to
`RELEASE_PLEASE_TOKEN`, then `GITHUB_TOKEN`. The latter suppresses downstream
workflow triggers; a PAT ties automation to a person's access and lifecycle.

Use a dedicated GitHub App with Contents write and Pull requests write. No
webhook service, OAuth user login, or long-running daemon is needed: Actions
mints a short-lived installation token per job and revokes it at job end.
Limit each token to the repository running the job. Metadata sync only needs
Contents write. Keep human review and branch protection in place.

Implementation sequence:

1. Supply a reusable App manifest and local manifest registration helper.
2. Register under `headroomlabs-ai` if an org owner can do so; otherwise use
   `JerrettDavis` and install the public App on selected repositories.
3. Store the returned private key as `RELEASE_APP_PRIVATE_KEY` and App ID as
   `RELEASE_APP_ID` using `gh` stdin; never write the key to disk or logs.
4. Replace both writers' fallback chains with required App authentication.
5. Test registration validation and workflow permissions, then qualify on
   the fork before adopting upstream.

The organization membership inspected on 2026-10-14 is `member`, and repository
permissions are push/triage without admin. The org currently has the PAT
secret and no App ID variable. Repository write access does not prove permission
to register an organization App or install it. Personal registration is the
available prototype path; an org owner must approve its installation upstream.

## Register and install

Authenticate `gh` as the intended App owner and sign in to GitHub as that same
owner in the browser before submitting the registration form, then run:

```powershell
python scripts/register-release-app.py --owner JerrettDavis --repo JerrettDavis/headroom
```

The helper serves a form only on loopback, opens the GitHub registration flow,
validates its callback state, exchanges the one-time code, and writes the two
Actions settings to the specified repository. It then opens the App's installation
page. Select **Only select repositories** and select `headroom`. Do not interrupt
the helper until it reports success. If repository configuration fails after
registration, the helper does not claim success: recover from the App settings
by generating a new private key, set the secret with `gh secret set
RELEASE_APP_PRIVATE_KEY --repo OWNER/headroom < key.pem`, and set the ID with
`gh variable set RELEASE_APP_ID --repo OWNER/headroom --body APP_ID`.
Securely remove the downloaded key after uploading it.

For an organization-owned registration, use `--owner headroomlabs-ai --organization`
and the appropriate `--repo`. This requires organization authorization. The public
personal App can also be installed in the org; ownership transfer can be handled
later in GitHub App settings.

## Qualification and rollout

On the fork, merge this workflow change after App installation and settings are
present. Push a conventional commit to `main`. Confirm the release workflow
mints an App token, a release PR is created by the App, and its CI starts.
Confirm metadata sync pushes a generated-file change and CI runs on that commit.
Then merge a fork release PR and verify a GitHub release and downstream release
and Docker runs are created. Fork publishing may require separate registry
credentials; do not repoint production registry credentials for this test.

For upstream, have an org owner install the App on `headroom` and configure the
same settings before merging the workflow change. Verify the same PR/CI/release
event chain there before deleting `RELEASE_PLEASE_TOKEN`. Missing settings or
token-mint failures now stop the job visibly. Roll back by reverting the workflow
change if needed while the existing PAT is retained during qualification.

References:

- https://github.com/googleapis/release-please-action#github-credentials
- https://github.com/actions/create-github-app-token
- https://docs.github.com/en/apps/sharing-github-apps/registering-a-github-app-from-a-manifest
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#triggering-a-workflow-from-a-workflow
