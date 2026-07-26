# Installing BioHybrid RoboTracker on a Mac

Written for someone who has never used GitHub, Python, Homebrew or the Terminal.
You need none of them. This is a download, a drag, and one unusual click.

**Time:** about ten minutes, most of it downloading.
**Download size:** roughly 450 MB.
**You do not need:** a GitHub account, Python, Homebrew, the command line, or
administrator rights.

---

## Before you start: will it run on your Mac?

Two things have to be true. Both take fifteen seconds to check.

**1. Your Mac must have Apple Silicon** — an M1, M2, M3, M4 or later chip. Click
the  menu in the top-left corner → **About This Mac**. Look at the line
labeled **Chip**.

- If it says *Apple M1*, *Apple M2 Pro*, *Apple M4 Max* or similar — you are fine.
- If it says **Intel** anything, **this application will not run on your Mac at
  all.** It is built as an Apple Silicon program, and an Intel Mac cannot execute
  one. Rosetta does not help — Rosetta translates Intel software to run on Apple
  Silicon, which is the opposite direction. Use the Windows version on a PC
  instead, or ask for an Intel build to be added.

**2. Your macOS must be version 12 (Monterey) or later.** Same **About This Mac**
window, the line above the chip. macOS 12.3 or later is better — that is the
version where Apple's GPU acceleration became available, and without it the
analysis falls back to the CPU and runs many times slower. It still *works*, it
is just slow.

---

## Step 1 — Download the installer

1. Open this page in Safari or Chrome:

   **https://github.com/Fettucciny/Biohybrid-Robot-Tracker/releases/latest**

   You do not need to sign in. GitHub is where the program is published; you are
   only downloading a file from it, the same as from any website.

2. Scroll down to a section headed **Assets**. You will see several files. You
   want the one ending in **`.dmg`** — its name looks like:

   ```
   robotrack-0.11.0-macos.dmg
   ```

   (The number changes with each version. The file is still called `robotrack`
   for technical reasons — it is the right file.)

3. Click it. The download starts. It is around 450 MB, so give it a minute or
   two. The program is large because everything it needs is packed inside it,
   which is precisely why you do not have to install Python or anything else.

4. Ignore the other files. `.exe` is the Windows version. `.zip` files are for
   people editing the source code.

---

## Step 2 — Open the downloaded file

Open your **Downloads** folder and double-click the `.dmg` file.

A `.dmg` is a *disk image* — macOS treats it like a USB stick you have just
plugged in. A window opens showing what is inside, and a small white drive icon
appears on your desktop and in the Finder sidebar.

**You have not installed anything yet.** This is the equivalent of opening the
box, not of putting the thing away.

---

## Step 3 — Drag the app into Applications

The window that opened shows two icons:

- **BioHybrid RoboTracker** on the left (named **robotrack** in version 0.10.0)
- a folder called **Applications** on the right

Drag the app icon onto the **Applications** folder and let go. A progress bar
appears for a few seconds while it copies.

That is the installation. macOS has no installer wizard for this kind of program
— putting the app in the Applications folder *is* installing it.

---

## Step 4 — Eject the disk image

Back in the Finder sidebar, find the white drive icon named **BioHybrid
RoboTracker** and click the **⏏** eject symbol next to it. (Or drag the desktop
icon to the Trash — for a disk image, that ejects rather than deletes.)

You can now delete the `.dmg` from Downloads. The app is copied, not linked; the
`.dmg` was only the delivery box.

---

## Step 5 — The first launch, which macOS will refuse

**This is the only genuinely confusing step, and it is expected.** Read it before
you try, because the first thing macOS tells you is wrong.

Open your **Applications** folder and double-click **BioHybrid RoboTracker**. One
of these two things happens.

### If you see "cannot be opened because the developer cannot be verified"

1. Close the dialog by clicking **Cancel** — *not* "Move to Trash".
2. In Applications, **right-click** (or hold **Control** and click) the app icon.
3. Choose **Open** from the menu that appears.
4. A near-identical dialog appears, but this one has an **Open** button. Click it.

The app starts, and macOS remembers your decision permanently. Every launch from
now on is an ordinary double-click.

### If nothing happens, or the dialog has no Open option

Newer macOS versions (15 Sequoia and later) moved this. Do the following instead:

1. Double-click the app once and dismiss whatever appears.
2. Open **System Settings** →  **Privacy & Security**.
3. Scroll down to the **Security** section. There is a line saying *"BioHybrid
   RoboTracker was blocked to protect your Mac"* with an **Open Anyway** button.
4. Click **Open Anyway**, authenticate with your password or Touch ID, then
   launch the app again.

### Why this happens, so you can judge it for yourself

Apple charges an annual fee for a Developer ID certificate and a notarization
service. This application is signed, but not with a paid certificate, so macOS
cannot look up who published it. The warning means *"I do not know who made
this"* — not *"I found something wrong with it."* macOS shows the same warning
for every unsigned program, including a great deal of academic software.

You are choosing to trust the person who sent you this link. That is the actual
decision, and no dialog can make it for you.

---

## Step 6 — Check it is using the GPU

When the app opens you get a splash screen for a few seconds while it loads, then
the main window.

Look at the **top-right corner of the window**. There is a small chip-shaped
label showing which processor it will use for analysis. It should name your
Apple chip — *Apple M2 Pro*, for instance.

If it says **CPU only**, the analysis will still be correct but several times
slower. That happens on macOS older than 12.3. There is nothing to fix in the
app; updating macOS is the fix.

---

## Step 7 — Your first analysis

1. **Video** → choose a recording. The panel on the right immediately lists every
   other video in the same folder, so you only do this once per session.
2. **CAD outline** → choose your `.DXF` drawing, and set the **scale** if the
   drawing is not already in real millimeters.
3. **Output folder** → choose where results should go. Each video gets its own
   subfolder inside it, named after the video.
4. Press **Run analysis** above the video.

A chime plays when it finishes; a lower, falling tone means it stopped early or
failed. Results — a `.csv`, plots, and an overlay video — land in the output
folder.

Every control in the sidebar has a small **?** next to it that explains what the
setting does and when to change it. Nothing in this program expects you to guess.

---

## Keeping it up to date

Press **Update** in the top-right corner of the window. It checks GitHub and
installs anything newer.

Updates are normally a few hundred kilobytes and apply in about a second — the
450 MB download happens once. When a new version is waiting, the Update button
pulses gently rather than interrupting you.

If it ever says no update channel is configured, open **Settings** and set the
channel to exactly:

```
github:Fettucciny/Biohybrid-Robot-Tracker
```

---

## When something goes wrong

### "The application is damaged and can't be opened. You should move it to the Trash."

Nothing is damaged. macOS tags files downloaded from the internet with an
invisible "quarantine" marker, and that marker plus an unpaid certificate
occasionally produces this misleading message instead of the normal one.

Clearing the marker takes one command. If you have never used Terminal, this is
genuinely safe and takes thirty seconds:

1. Press **⌘ Space**, type `Terminal`, press **Return**. A window with a text
   prompt opens.
2. Type this, **including the trailing space**, but do not press Return yet:

   ```
   xattr -dr com.apple.quarantine 
   ```

3. Now open your Applications folder and **drag the BioHybrid RoboTracker icon
   into the Terminal window**. Terminal fills in the full path for you — which
   saves typing it, and avoids the mistakes that spaces in names cause.
4. *Now* press **Return**. If it prints nothing at all, it worked. Silence is
   success in Terminal.
5. Launch the app again.

What the command does: `xattr` manages the hidden tags on a file, `-d` deletes
one, `-r` applies it to everything inside the app, and
`com.apple.quarantine` is the tag's name. It changes nothing else.

### The icon bounces once in the Dock and then disappears

The program failed while starting. It writes a log explaining why. To find it:

1. In Finder, press **⇧⌘G** (Shift-Command-G).
2. Paste this and press Return:

   ```
   ~/Library/Application Support/robotrack
   ```

3. Open `robotrack-error.log` or `robotrack-selftest.log` in TextEdit and send
   the contents to whoever gave you the app. The last few lines name the problem.

### "You do not have permission to open the application"

The app landed somewhere it cannot run from — usually still inside the mounted
disk image. Confirm it is in **Applications**, not on the white drive icon, and
that you ejected the disk image in Step 4.

### Analysis is extremely slow

Check the chip label from Step 6. If it says **CPU only**, see Step 6. If it
names your Apple chip and it is still slow, 4K footage is simply heavy work — try
setting the **decode scale** to 0.5 in the sidebar, which halves the picture size
and roughly quarters the work, at some cost in precision.

### It cannot find the video, or a decode fails

Videos stored in iCloud Drive are sometimes not really on your Mac — only a
placeholder is. In Finder, a cloud icon next to the filename means the file is
still in the cloud. Right-click it and choose **Download Now**, wait for the
cloud icon to disappear, and try again.

---

## Uninstalling

Drag **BioHybrid RoboTracker** from Applications to the Trash. That removes the
program entirely.

If you also want the settings gone, delete this folder (⇧⌘G in Finder to reach
it):

```
~/Library/Application Support/robotrack
```

Your videos and your results are untouched by either step — they were never
inside the app.

---

## Appendix — running from source instead

**You almost certainly do not want this section.** It exists for someone who
wants to modify the code. Installing the `.dmg` above gives you the identical
program with none of the following.

Running from source means installing Python and the libraries the program uses,
which is what Homebrew and the setup script are for:

1. Install Homebrew — a package manager for macOS — by pasting the command from
   **https://brew.sh** into Terminal.
2. Get the source: on the repository page, **Code → Download ZIP**, then unzip it.
3. In Terminal, `cd` into that folder and run:

   ```bash
   ./install_macos.sh
   ```

   That installs ffmpeg through Homebrew, creates an isolated Python
   environment, and installs the program into it. It prints whether GPU
   acceleration is available at the end.

4. To start it afterwards:

   ```bash
   source .venv/bin/activate
   robotrack-gui
   ```

Note that on macOS the plain `pip install torch` is the correct one — there is no
CUDA index to choose, as there is on Windows. The Mac wheel already contains
Apple's Metal support.

To build your own `.dmg` from source, run `./build_macos.sh`. It takes twenty
minutes or so and produces exactly what you downloaded in Step 1.
