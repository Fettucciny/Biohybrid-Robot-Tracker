# Publishing BioHybrid RoboTracker, and installing it

**Part 1** is you, once, and it assumes you have never used git before —
everything is spelled out. **Part 2** is text you can send to anyone who needs
to run the software.

---

# Part 0 — What you are actually about to do

Five words do all the work, and they will make more sense if you meet them
before you type them.

**Repository** ("repo"). A folder that GitHub keeps a copy of, with its full
history. Yours will be a website at `github.com/Fettucciny/Biohybrid-Robot-Tracker`.

**Commit.** A saved snapshot of the folder, with a note attached. Nothing is
sent anywhere yet — a commit is local, on your own disk.

**Push.** Upload your commits to GitHub. This is the step that makes anything
public.

**Tag.** A label on one particular commit, like `v0.10.0`. GitHub notices when a
tag arrives and this is what starts the automatic build.

**Actions.** GitHub's build robots. When your tag lands, they check out your
code onto a fresh Windows machine and a fresh Mac, build the installer on each,
and attach both to a Release page. This is why you never have to own a Mac.

The whole sequence is: *commit* your files → *push* them to a new *repo* → push
a *tag* → *Actions* builds the installers → people download them.

You will do Part 1 once. After that, publishing a new version is three
commands, listed at the end.

---

# Part 1 — Get it on GitHub

## Step 1. Install the two tools you need

Open **PowerShell**: press the Windows key, type `powershell`, press Enter.
Paste these one at a time:

```powershell
winget install --id Git.Git -e
winget install --id GitHub.cli -e
```

`git` is the version-control program itself. `gh` is GitHub's own command-line
tool — it is not strictly required, but it creates the repository and logs you
in without you having to click through the website, which removes most of the
places a first attempt goes wrong.

**Now close PowerShell and open a new one.** Installers add things to your PATH,
and an already-open window does not notice. This trips up nearly everyone once.

Check both are there:

```powershell
git --version
gh --version
```

You want two version numbers. If you get *"not recognized as the name of a
cmdlet"*, the new-window step did not happen, or the install failed — scroll up
in the install output to see which.

## Step 2. Tell git who you are

Git stamps every commit with a name and an email. It refuses to commit until it
knows them, and the error it gives is confusing if you have not seen it before.
Set them once, for all projects:

```powershell
git config --global user.name "Yikang Xu"
git config --global user.email "<your GitHub no-reply address>"
```

**Use GitHub's no-reply address, not your real one.** Every commit carries its
author's email forever, in public, where it is trivially scraped. GitHub gives
you a permanent alias that still links commits to your account: go to
**https://github.com/settings/emails** and read the address shown under *"Keep
my email addresses private"*. It looks like
`12345678+Fettucciny@users.noreply.github.com` — copy it exactly, the leading
number is your account ID.

If you skip this and GitHub's *"Block command line pushes that expose my email"*
setting is on, the push is rejected at the very end with error `GH007`. The fix
after the fact is in the troubleshooting table below.

## Step 3. Go to the project folder

Everything from here happens **inside your project folder**, not in whatever
folder PowerShell happens to open in. Move there:

```powershell
cd "C:\Users\Yikan\Nextcloud2\BioHybrid Tracker"
```

The quotes matter — there is a space in "BioHybrid Tracker", and without quotes
PowerShell reads it as two separate things.

Check you are in the right place:

```powershell
ls
```

You should see `robotrack`, `launcher`, `tools`, `tests`, `README.md`,
`LICENSE`, `build_exe.ps1` and friends. If you do not, you are in the wrong
folder and nothing after this will work.

> **A note on Nextcloud.** This folder syncs to the cloud, and git is about to
> create a hidden `.git` folder inside it with thousands of small files.
> Nextcloud will sync those too. It works, and plenty of people do it — but if
> Nextcloud ever reports conflicts inside `.git`, that is why, and the fix is to
> add `.git` to Nextcloud's ignore list in its settings.

## Step 4. Put the build workflow where GitHub looks for it

GitHub only runs build instructions that live in a folder called
`.github\workflows`. The file is currently sitting in `tools\` instead, because
the tool that synced it to your machine is not permitted to write to `.github\`.
Move a copy into place:

```powershell
New-Item -ItemType Directory -Force .github\workflows | Out-Null
Copy-Item tools\release-workflow.yml .github\workflows\release.yml
```

The first line creates the folder (`-Force` means "don't complain if it already
exists"; `| Out-Null` just hides the confirmation message). The second copies
the file in and renames it.

Confirm it landed:

```powershell
ls .github\workflows
```

You want to see `release.yml`.

## Step 5. Turn the folder into a repository

```powershell
git init -b main
```

This creates the hidden `.git` folder. Nothing has been sent anywhere; you have
just told git to start paying attention to this directory. `-b main` names the
starting branch `main`, which is what GitHub expects.

## Step 6. Choose what goes in, and look at it before you commit

```powershell
git add .
```

`git add .` means "stage everything in this folder". *Staged* means "included in
the next snapshot". This does not upload anything.

Now look at what you just staged — **this is the one step worth slowing down
for**:

```powershell
git status
```

You will get a long list under *"Changes to be committed"*. What you want to see
is roughly:

- `robotrack/` — about 25 `.py` files
- `launcher/`, `tools/`, `tests/`
- `README.md`, `PUBLISHING.md`, `LICENSE`, `pyproject.toml`
- `build_exe.ps1`, `build_macos.sh`, `install_windows.ps1`, `install_macos.sh`
- `.github/workflows/release.yml`

What you must **not** see:

- `.venv/` — a copy of Python, hundreds of megabytes
- `build/` or `dist/` — the built application, several gigabytes
- `launcher/bin/` — ffmpeg, about 200 MB
- `updates/` — the update packages
- `TestRun/`, `IMG_3076_cache.mov`, or anything ending `.MOV`

None of those should appear, because `.gitignore` already excludes them. The
check matters because GitHub refuses any single file over 100 MB, and it refuses
it at the *end* of a long upload rather than the start.

If something large did slip through, remove it from the snapshot without
deleting it from your disk:

```powershell
git rm --cached -r .venv
```

(substituting whatever appeared), then run `git status` again.

When the list looks right:

```powershell
git commit -m "BioHybrid RoboTracker: GPU-accelerated tracking for muscle-driven soft robots"
```

The text after `-m` is the note attached to the snapshot. Git prints something
like *"120 files changed, 15000 insertions(+)"*. That is your first commit —
still entirely on your own machine.

## Step 7. Log in to GitHub

```powershell
gh auth login
```

This asks a short series of questions with arrow-key menus. The answers:

| question | answer |
| --- | --- |
| What account do you want to log into? | **GitHub.com** |
| What is your preferred protocol for Git operations? | **HTTPS** |
| Authenticate Git with your GitHub credentials? | **Yes** |
| How would you like to authenticate? | **Login with a web browser** |

It then shows an eight-character code like `A1B2-C3D4`. Copy it, press Enter,
and your browser opens to a GitHub page asking for that code. Paste it,
authorise, and go back to PowerShell — it will say *"Logged in as Fettucciny"*.

If you do not have a GitHub account yet, make one at github.com first; it is
free and takes a minute.

## Step 8. Connect to the repository and upload

The repository already exists at
`github.com/Fettucciny/Biohybrid-Robot-Tracker`, so this step is *connect and
upload*, not create. Do **not** run `gh repo create` — that command makes a new
repository and fails with *"Name already exists on this account"* when one is
already there.

```powershell
git remote add origin https://github.com/Fettucciny/Biohybrid-Robot-Tracker.git
git push -u origin main
```

`git remote add origin <url>` writes GitHub's address into your local repo under
the nickname `origin`; nothing is transferred yet. `git push -u origin main`
does the upload, and the `-u` makes `origin main` the default so that every
later push is just `git push`.

Open the repository in a browser — your files should all be there, with the
README displayed underneath them.

### If the push is rejected

If you created the repository with "Add a README", a license or a `.gitignore`
ticked, GitHub already put a commit in it that your local copy has never seen,
and git refuses to overwrite history it does not recognize:

```
! [rejected]  main -> main (fetch first)
error: failed to push some refs
```

Those starter files are auto-generated and you have better versions of all three
already, so the fix is to tell git that your copy is the authoritative one:

```powershell
git push -u origin main --force
```

`--force` is worth being careful with in general — it discards whatever was on
the remote. Here that is nothing but GitHub's placeholder commit, which is why
it is safe *this once*. Do not make a habit of it on a repository other people
are also pushing to.

### If you ever rename your GitHub account

GitHub redirects the old URL to the new one, so things keep working — right up
until someone else registers your old username, at which point every redirect
breaks at once. Treat the redirect as a grace period, not a solution, and update
these three:

```powershell
git remote set-url origin https://github.com/Fettucciny/Biohybrid-Robot-Tracker.git
git remote -v                        # confirm both lines say Fettucciny
```

...the update channel in the app's Settings (`github:Fettucciny/Biohybrid-Robot-Tracker`),
and the no-reply address from Step 2, which contains your username. Read the new
one off **https://github.com/settings/emails** and set it with
`git config --global user.email "..."`. Commits already made with the old form
still attribute correctly — the numeric ID in front of the `+` is what GitHub
matches on, and that never changes — so there is no need to rewrite history.

### The repository has to be public

A private repository will look fine to you and be broken for everyone else. The
**Update** button fetches releases over the plain GitHub API and sends no
password or token with the request, so against a private repository it simply
gets "not found" — and so does anyone following a download link, including you
when logged out.

Check on the repository page: if there is a **Private** badge next to the name,
go to **Settings** → **General** → scroll to the bottom → **Change repository
visibility** → **Make public**.

## Step 9. Let the build robots write releases

On your repository page: **Settings** → **Actions** (left sidebar) → **General**
→ scroll to **Workflow permissions** → select **Read and write permissions** →
**Save**.

Do not skip this. Without it, the build runs happily for half an hour and then
fails on its very last action with a `403` error, because it is not allowed to
create the Release page.

## Step 10. Publish the first release

A release is triggered by pushing a tag. The tag has to match the version inside
the code exactly — that is already `0.10.0`, so:

```powershell
git tag v0.10.0
git push origin v0.10.0
```

The first line creates the label locally; the second sends it to GitHub, which
is what the build robots are watching for.

Now go to the **Actions** tab on your repository page. You will see a run
appear, with jobs called *version agrees with tag*, *code patch*, *windows
installer*, *macos bundle*, and *publish release*. The version check finishes in
seconds; Windows and macOS build in parallel and take roughly 25–35 minutes,
most of it spent downloading PyTorch. Green ticks mean success; a red X is
clickable and shows the log.

When it finishes, look at the **Releases** section on the right of your
repository's front page. You will have:

| asset | who needs it |
| --- | --- |
| `robotrack-0.10.0-setup.exe` | Windows users |
| `robotrack-0.10.0-macos.dmg` | Mac users |
| `robotrack-0.10.0-code.zip` | nobody directly — the **Update** button uses this |

That link — `github.com/Fettucciny/Biohybrid-Robot-Tracker/releases/latest` — is what you
send people.

## Step 11. Point the app at the repository

Open robotrack, go to **Settings**, and set the **update channel** to:

```
github:Fettucciny/Biohybrid-Robot-Tracker
```

(the literal word `github`, a colon, then your username and repo name — no
`https://`).

That single string replaces the Nextcloud folder you have been using. From now
on the **Update** button reads GitHub directly and picks the right file for
whatever machine it is running on. Anyone who installs the app should set the
same string once.

---

## Publishing later versions

Once Part 1 is done, a new release is three commands. Say you have made changes
and want to ship 0.11.0:

```powershell
# 1. Open robotrack\__init__.py and change __version__ to "0.11.0". Save.

# 2. Snapshot and upload the changes
git commit -am "v0.11.0"
git push

# 3. Tag it, which starts the build
git tag v0.11.0
git push origin v0.11.0
```

`git commit -am` is `git add` and `git commit` combined, for files git already
knows about. If you have *added* a brand-new file, use `git add .` first.

**The tag and `__version__` must match.** The first job in the workflow does
nothing but compare them and fails the whole run if they disagree. That guard is
there because the mistake it catches is silent and unpleasant: a release
labeled 0.11.0 whose contents report 0.10.0 leaves every installed copy
convinced it is already up to date, and nobody notices until someone asks why a
fix never arrived.

## When something goes wrong

| what you see | what it means |
| --- | --- |
| `fatal: not a git repository` | You are not in the project folder. Re-run the `cd` from Step 3. |
| `Please tell me who you are` | Step 2 was skipped. |
| `git: command not found` / `not recognized` | Open a new PowerShell window (Step 1). |
| `remote: Permission to ... denied` | `gh auth login` did not complete. Re-run Step 7. |
| The `publish release` job fails with `403` | Step 9 was skipped. Fix the setting, then delete the tag and re-push it: `git tag -d v0.10.0; git push origin :v0.10.0; git tag v0.10.0; git push origin v0.10.0` |
| `Tag v0.11.0 does not match __version__` | You tagged without bumping the version, or bumped without committing. |
| `file is 130.00 MB; exceeds GitHub's limit` | Something excluded got committed. `git rm --cached` it, commit again. |
| `GH007: Your push would publish a private email address` | Your commits are stamped with your real email and GitHub is set to block that. Set the no-reply address from Step 2, then restamp the commit you already made: `git config --global user.email "<no-reply address>"` then `git commit --amend --reset-author --no-edit` then push again. |

---

# Part 2 — What to send other people

Everything below is written to be forwarded as-is.

> ## Installing BioHybrid RoboTracker
>
> Go to **https://github.com/Fettucciny/Biohybrid-Robot-Tracker/releases/latest** and
> download the file for your machine. You do not need Python, CUDA, ffmpeg or
> anything else installed first — it is all inside.
>
> ### Windows
>
> 1. Download `robotrack-<version>-setup.exe`.
> 2. Run it. Windows will show a blue **"Windows protected your PC"** panel —
>    click **More info**, then **Run anyway**. That warning means the installer
>    has no paid code-signing certificate, not that anything was found in it.
> 3. It installs for your user only, so it needs no administrator rights.
> 4. Launch **BioHybrid RoboTracker** from the Start menu.
>
> ### macOS (Apple Silicon — M1 and later)
>
> There is a longer, click-by-click version of this in **MACOS-INSTALL.md** in
> the repository, written for someone who has never installed anything this way.
> Send that link instead if the four steps below are not enough.
>
> 1. Download `robotrack-<version>-macos.dmg`.
> 2. Open it and drag **BioHybrid RoboTracker** onto the **Applications** folder.
> 3. The first launch will be refused: *"robotrack cannot be opened because the
>    developer cannot be verified."* Right-click the app in Applications, choose
>    **Open**, then **Open** again in the dialog. macOS remembers this
>    permanently — you only do it once.
>
>    If you would rather clear it in one command, in Terminal:
>
>    ```bash
>    xattr -dr com.apple.quarantine "/Applications/BioHybrid RoboTracker.app"
>    ```
>
>    Both routes do the same thing. The app is signed, just not with a paid
>    Apple Developer ID, and this is Gatekeeper's standard response to that.
>
> ### Keeping it current
>
> The **Update** button in the header does everything. If it says no channel is
> configured, open **Settings** and set the update channel to:
>
> ```
> github:Fettucciny/Biohybrid-Robot-Tracker
> ```
>
> Updates are normally a few hundred kilobytes and apply in about a second — the
> multi-gigabyte download only happens on a first install or when a dependency
> changes.

---

# Installing with no internet

The installers are ordinary files with no network dependency, so a flash drive
or a synced share works exactly as well as GitHub — useful for an offline rig or
a machine behind a restrictive proxy.

1. Download all three release assets onto the drive, into one folder.
2. Once, from a machine with the source checked out, write a manifest beside
   them so the folder also works as an update channel:

   ```powershell
   python tools\publish_update.py E:\robotrack-updates
   ```

3. On each target machine: run the `.exe` or the `.dmg`, then set the app's
   update channel to that folder (`E:\robotrack-updates`, or the UNC path of the
   share). Patching then works off the drive with no GitHub access at all.
