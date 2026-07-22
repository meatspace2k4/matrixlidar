#!/usr/bin/env python3
"""branch_harness.py: sequential, resumable branch->test->PR->merge->regression workflow.

Codifies the local development workflow for this repo (no remote; the "PR" is a
local review record under .harness/, not a GitHub push). State persists in
.harness/state.json so an agent can execute the workflow step by step across
invocations. Steps that need information the script cannot know are "mad-lib
blanks": `next` refuses to run a step until its blanks are filled via `fill`.

Steps, in order:
  1. checkpoint   record the base branch + commit (current HEAD).
  2. branch       create + switch to the work branch.        [BRANCH_NAME]
  3. implement    pause-point; record what was implemented.  [IMPLEMENTATION_SUMMARY]
  4. tests        record test files, run `python3 tests/run.py`;
                  failing suite blocks progression.          [TEST_FILES]
  5. pr-open      write the local PR record (.harness/pr.json) capturing
                  base/work branches, shas, test result.     [PR_TITLE, PR_DESCRIPTION]
  6. pr-review    record verdict; reject rolls the workflow back to the tests
                  step (resumable after more commits).       [REVIEW_VERDICT, REVIEW_NOTES?]
  7. pr-close     merge work branch into base (--no-ff), record merge commit.
  8. regression   run the suite on the work branch, then the merged base,
                  restore the original branch, report both results.

Blanks:
  BRANCH_NAME             work branch to create
  IMPLEMENTATION_SUMMARY  free-text summary of the coding work done
  TEST_FILES              test files added/changed (free text)
  PR_TITLE                PR record title
  PR_DESCRIPTION          PR record description
  REVIEW_VERDICT          approve | reject
  REVIEW_NOTES            optional reviewer notes

Commands:
  dry-run   print the full ordered plan with {{BLANK: description}} placeholders;
            executes nothing.
  status    show state, filled blanks, pending step. Always safe.
  next      execute the next pending step. Missing blank: prints the {{...}}
            prompt and exits 2 without doing anything. Failed step: exits
            nonzero, state unchanged, re-running `next` retries it.
  fill      supply a blank: fill BLANK_NAME "value". Always safe.
  reset     delete state; refuses if the work branch was already created
            unless --force.

Exit codes: 0 success, 2 missing blank, 1 anything else.

Example agent session:
  scripts/branch_harness.py dry-run
  scripts/branch_harness.py next                     # checkpoint
  scripts/branch_harness.py next                     # exits 2: needs BRANCH_NAME
  scripts/branch_harness.py fill BRANCH_NAME fix-ghost-returns
  scripts/branch_harness.py next                     # branch created
  ... do the implementation work, commit it ...
  scripts/branch_harness.py fill IMPLEMENTATION_SUMMARY "Clamp ghost returns in doorcal"
  scripts/branch_harness.py next                     # implement
  scripts/branch_harness.py fill TEST_FILES "tests/test_doorcal.py"
  scripts/branch_harness.py next                     # tests (runs the suite)
  scripts/branch_harness.py fill PR_TITLE "Fix ghost returns"
  scripts/branch_harness.py fill PR_DESCRIPTION "Clamps returns below floor plane."
  scripts/branch_harness.py next                     # pr-open
  scripts/branch_harness.py fill REVIEW_VERDICT approve
  scripts/branch_harness.py next                     # pr-review
  scripts/branch_harness.py next                     # pr-close (merge --no-ff)
  scripts/branch_harness.py next                     # regression (both branches)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_DIR = os.path.join(REPO, ".harness")
STATE_PATH = os.path.join(HARNESS_DIR, "state.json")
PR_PATH = os.path.join(HARNESS_DIR, "pr.json")
TEST_CMD = ["python3", "tests/run.py"]

BLANKS = {
    "BRANCH_NAME": "name of the work branch to create",
    "IMPLEMENTATION_SUMMARY": "free-text summary of the implementation work done on the work branch",
    "TEST_FILES": "test files added or changed (free text, e.g. comma-separated paths)",
    "PR_TITLE": "title for the local PR record",
    "PR_DESCRIPTION": "description for the local PR record",
    "REVIEW_VERDICT": "review verdict: approve or reject",
    "REVIEW_NOTES": "optional reviewer notes",
}

# (name, required blanks, description). REVIEW_NOTES is optional and never required.
STEPS = [
    ("checkpoint", [], "record base branch + commit (current HEAD)"),
    ("branch", ["BRANCH_NAME"], "create + switch to the work branch"),
    ("implement", ["IMPLEMENTATION_SUMMARY"],
     "pause-point: record the implementation summary (coding happens outside the script)"),
    ("tests", ["TEST_FILES"],
     "record test files, run `python3 tests/run.py`; failure blocks progression"),
    ("pr-open", ["PR_TITLE", "PR_DESCRIPTION"],
     "write local PR record (.harness/pr.json): branches, shas, test result"),
    ("pr-review", ["REVIEW_VERDICT"],
     "record verdict; reject rolls back to the tests step (REVIEW_NOTES optional)"),
    ("pr-close", [], "merge work branch into base with --no-ff, record merge commit"),
    ("regression", [],
     "run suite on work branch then merged base, restore original branch, report both"),
]

STEP_NAMES = [s[0] for s in STEPS]


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def git(*args, check=True):
    p = subprocess.run(["git", "-C", REPO] + list(args),
                       capture_output=True, text=True)
    if check and p.returncode != 0:
        print("git %s failed (exit %d):" % (" ".join(args), p.returncode),
              file=sys.stderr)
        sys.stderr.write(p.stderr)
        sys.exit(1)
    return p


def load_state():
    if not os.path.exists(STATE_PATH):
        return {"done": [], "blanks": {}, "branch_created": False, "results": {}}
    with open(STATE_PATH) as f:
        return json.load(f)


def save_state(state):
    os.makedirs(HARNESS_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")
    os.replace(tmp, STATE_PATH)


def save_pr(record):
    os.makedirs(HARNESS_DIR, exist_ok=True)
    tmp = PR_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    os.replace(tmp, PR_PATH)


def load_pr():
    if not os.path.exists(PR_PATH):
        return None
    with open(PR_PATH) as f:
        return json.load(f)


def pending_step(state):
    for name in STEP_NAMES:
        if name not in state["done"]:
            return name
    return None


def blank_prompt(name):
    return "{{%s: %s}}" % (name, BLANKS[name])


def working_tree_dirty():
    # Tracked changes only; untracked files (.harness/, new test fixtures
    # pre-add) do not block.
    p = git("status", "--porcelain", "--untracked-files=no")
    return bool(p.stdout.strip())


def current_branch():
    return git("branch", "--show-current").stdout.strip()


def head_sha():
    return git("rev-parse", "HEAD").stdout.strip()


def run_suite():
    """Run the test suite from the repo root; return (passed, returncode)."""
    print("running: %s (cwd=%s)" % (" ".join(TEST_CMD), REPO))
    p = subprocess.run(TEST_CMD, cwd=REPO)
    return p.returncode == 0, p.returncode


# ---------------------------------------------------------------- steps

def step_checkpoint(state):
    branch = current_branch()
    if not branch:
        die("checkpoint: detached HEAD; check out a branch first")
    state["base_branch"] = branch
    state["base_commit"] = head_sha()
    print("checkpoint: base branch %s at %s" % (branch, state["base_commit"][:12]))


def step_branch(state):
    name = state["blanks"]["BRANCH_NAME"]
    exists = git("rev-parse", "--verify", "--quiet",
                 "refs/heads/" + name, check=False).returncode == 0
    if exists:
        # Retry after a partial earlier attempt: just switch to it.
        git("checkout", name)
        print("branch: %s already existed, switched to it" % name)
    else:
        git("checkout", "-b", name, state["base_commit"])
        print("branch: created and switched to %s from %s"
              % (name, state["base_commit"][:12]))
    state["branch_created"] = True


def step_implement(state):
    summary = state["blanks"]["IMPLEMENTATION_SUMMARY"]
    state["results"]["implementation_summary"] = summary
    print("implement: recorded summary (%d chars)" % len(summary))


def step_tests(state):
    files = state["blanks"]["TEST_FILES"]
    passed, rc = run_suite()
    state["results"]["tests"] = {
        "files": files,
        "commit": head_sha(),
        "passed": passed,
    }
    if not passed:
        # Persist the failing result but do not mark the step done.
        save_state(state)
        die("tests: suite FAILED (exit %d). Fix, commit, re-run `next`." % rc)
    print("tests: suite passed at %s (files: %s)"
          % (head_sha()[:12], files))


def step_pr_open(state):
    tests = state["results"].get("tests")
    prev = load_pr()
    record = {
        "title": state["blanks"]["PR_TITLE"],
        "description": state["blanks"]["PR_DESCRIPTION"],
        "base_branch": state["base_branch"],
        "base_commit": state["base_commit"],
        "work_branch": state["blanks"]["BRANCH_NAME"],
        "work_commit": head_sha(),
        "test_result": tests,
        "revision": (prev["revision"] + 1) if prev else 1,
        "reviews": prev["reviews"] if prev else [],
        "status": "open",
    }
    save_pr(record)
    print("pr-open: wrote %s (revision %d, %s -> %s)"
          % (PR_PATH, record["revision"],
             record["work_branch"], record["base_branch"]))


def step_pr_review(state):
    verdict = state["blanks"]["REVIEW_VERDICT"].strip().lower()
    notes = state["blanks"].get("REVIEW_NOTES", "")
    if verdict not in ("approve", "reject"):
        die("pr-review: REVIEW_VERDICT must be 'approve' or 'reject', got %r"
            % verdict, 2)
    pr = load_pr()
    if pr is None:
        die("pr-review: no PR record found; state is inconsistent")
    pr["reviews"].append({"verdict": verdict, "notes": notes,
                          "work_commit": head_sha()})
    if verdict == "reject":
        pr["status"] = "changes-requested"
        save_pr(pr)
        # Roll back to the tests step so the fixed code is re-tested and the
        # PR record refreshed; require a fresh verdict next time around.
        state["done"] = [s for s in state["done"]
                         if s not in ("tests", "pr-open")]
        state["blanks"].pop("REVIEW_VERDICT", None)
        state["blanks"].pop("REVIEW_NOTES", None)
        save_state(state)
        die("pr-review: REJECTED. %s\nWorkflow rolled back to the tests step; "
            "commit fixes, then re-run `next`." % (notes or "(no notes)"))
    pr["status"] = "approved"
    save_pr(pr)
    print("pr-review: approved. %s" % (notes or ""))


def step_pr_close(state):
    pr = load_pr()
    if pr is None or pr.get("status") != "approved":
        die("pr-close: PR record is not approved; state is inconsistent")
    base = state["base_branch"]
    work = state["blanks"]["BRANCH_NAME"]
    git("checkout", base)
    p = git("merge", "--no-ff", work,
            "-m", "Merge %s into %s: %s" % (work, base, pr["title"]),
            check=False)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        sys.stderr.write(p.stderr)
        die("pr-close: merge failed (exit %d). Resolve or `git merge --abort`, "
            "then re-run `next`." % p.returncode)
    merge_sha = head_sha()
    state["results"]["merge_commit"] = merge_sha
    pr["status"] = "merged"
    pr["merge_commit"] = merge_sha
    save_pr(pr)
    print("pr-close: merged %s into %s at %s" % (work, base, merge_sha[:12]))


def step_regression(state):
    base = state["base_branch"]
    work = state["blanks"]["BRANCH_NAME"]
    original = current_branch()
    results = {}
    try:
        for label, branch in (("work", work), ("base", base)):
            git("checkout", branch)
            passed, rc = run_suite()
            results[label] = {"branch": branch, "passed": passed, "exit": rc}
    finally:
        p = git("checkout", original, check=False)
        if p.returncode != 0:
            print("regression: FAILED to restore branch %r:" % original,
                  file=sys.stderr)
            sys.stderr.write(p.stderr)
    state["results"]["regression"] = results
    print("regression results:")
    for label in ("work", "base"):
        r = results.get(label)
        if r is None:
            print("  %-4s ?: not run" % label)
            continue
        print("  %-4s %-30s %s" % (label, "(%s):" % r["branch"],
                                   "PASS" if r["passed"] else
                                   "FAIL (exit %d)" % r["exit"]))
    if not all(r.get("passed") for r in results.values()) or len(results) < 2:
        save_state(state)
        die("regression: FAILURE detected. Workflow NOT complete; "
            "re-run `next` to retry after fixing.")


STEP_FUNCS = {
    "checkpoint": step_checkpoint,
    "branch": step_branch,
    "implement": step_implement,
    "tests": step_tests,
    "pr-open": step_pr_open,
    "pr-review": step_pr_review,
    "pr-close": step_pr_close,
    "regression": step_regression,
}

# ------------------------------------------------------------- commands

def cmd_dry_run(_args):
    print("branch_harness plan (repo: %s). Nothing below is executed." % REPO)
    print("State file: %s" % STATE_PATH)
    print()
    for i, (name, required, desc) in enumerate(STEPS, 1):
        print("%d. %s: %s" % (i, name, desc))
        for b in required:
            print("   needs %s" % blank_prompt(b))
        if name == "pr-review":
            print("   optional %s" % blank_prompt("REVIEW_NOTES"))
    print()
    print("Fill blanks with: branch_harness.py fill BLANK_NAME \"value\"")
    print("Advance with:     branch_harness.py next")


def cmd_status(_args):
    state = load_state()
    pend = pending_step(state)
    print("repo: %s" % REPO)
    print("state file: %s (%s)"
          % (STATE_PATH, "present" if os.path.exists(STATE_PATH) else "absent"))
    if state.get("base_branch"):
        print("base: %s @ %s" % (state["base_branch"], state["base_commit"][:12]))
    print("steps:")
    for name, required, _desc in STEPS:
        if name in state["done"]:
            mark = "done"
        elif name == pend:
            missing = [b for b in required if b not in state["blanks"]]
            mark = "PENDING" + (" (missing: %s)" % ", ".join(missing)
                                if missing else "")
        else:
            mark = "-"
        print("  %-10s %s" % (name, mark))
    if state["blanks"]:
        print("filled blanks:")
        for k, v in state["blanks"].items():
            print("  %s = %r" % (k, v))
    for key in ("tests", "merge_commit", "regression"):
        if key in state.get("results", {}):
            print("result %s: %s" % (key, json.dumps(state["results"][key])))
    if pend is None:
        print("workflow complete")


def cmd_fill(args):
    if args.blank not in BLANKS:
        die("unknown blank %r; valid: %s" % (args.blank, ", ".join(BLANKS)), 2)
    if args.blank == "REVIEW_VERDICT" and \
            args.value.strip().lower() not in ("approve", "reject"):
        die("REVIEW_VERDICT must be 'approve' or 'reject'", 2)
    state = load_state()
    state["blanks"][args.blank] = args.value
    save_state(state)
    print("filled %s" % args.blank)


def cmd_next(_args):
    state = load_state()
    name = pending_step(state)
    if name is None:
        print("workflow complete; nothing to do")
        return
    required = dict((s[0], s[1]) for s in STEPS)[name]
    missing = [b for b in required if b not in state["blanks"]]
    if missing:
        print("step %r needs blanks before it can run:" % name)
        for b in missing:
            print("  %s" % blank_prompt(b))
        print("Fill with: branch_harness.py fill BLANK_NAME \"value\"")
        sys.exit(2)
    if name != "checkpoint" and working_tree_dirty():
        die("working tree has uncommitted tracked changes; commit or stash "
            "before running step %r" % name)
    print("executing step: %s" % name)
    STEP_FUNCS[name](state)
    state["done"].append(name)
    save_state(state)
    nxt = pending_step(state)
    print("step %s done. next: %s" % (name, nxt or "(workflow complete)"))


def cmd_reset(args):
    state = load_state()
    if state.get("branch_created") and not args.force:
        die("reset refused: work branch %r was already created; "
            "use --force to reset anyway (branches are left untouched)"
            % state["blanks"].get("BRANCH_NAME"))
    if os.path.isdir(HARNESS_DIR):
        shutil.rmtree(HARNESS_DIR)
        print("removed %s" % HARNESS_DIR)
    else:
        print("no state to remove")


def main():
    ap = argparse.ArgumentParser(
        prog="branch_harness.py",
        description="Sequential, resumable branch->test->PR->merge->regression "
                    "workflow for this repo. See module docstring for the "
                    "step and blank lists.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dry-run", help="print the full plan with {{BLANK}} "
                                   "placeholders; executes nothing")
    sub.add_parser("status", help="show state, filled blanks, pending step")
    sub.add_parser("next", help="execute the next pending step "
                                "(exit 2 if a blank is missing)")
    p_fill = sub.add_parser("fill", help="supply a blank value")
    p_fill.add_argument("blank", metavar="BLANK_NAME")
    p_fill.add_argument("value")
    p_reset = sub.add_parser("reset", help="delete workflow state")
    p_reset.add_argument("--force", action="store_true",
                         help="reset even if the work branch was created")
    args = ap.parse_args()
    {"dry-run": cmd_dry_run, "status": cmd_status, "fill": cmd_fill,
     "next": cmd_next, "reset": cmd_reset}[args.cmd](args)


if __name__ == "__main__":
    main()
