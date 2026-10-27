"""25 verifiable bash tasks for terminal RL. Each verify is a shell script: exit 0 == success."""

TASKS = [
    dict(id="t01_mkdir", partial='test -d out', prompt="Create a directory named out/ and inside it a file result.txt whose only line is exactly: done",
         setup={}, verify='[ -f out/result.txt ] && [ "$(cat out/result.txt)" = "done" ]'),

    dict(id="t02_count_lines", partial='test -f answer.txt', prompt="The file data.txt contains several lines. Write the number of lines to answer.txt (just the integer).",
         setup={"data.txt": "alpha\nbeta\ngamma\ndelta\nepsilon\n"},
         verify='[ "$(tr -d " \n" < answer.txt)" = "5" ]'),

    dict(id="t03_grep", partial='test -f errors.txt', prompt="Extract only the lines containing the word ERROR from log.txt into errors.txt, preserving order.",
         setup={"log.txt": "INFO start\nERROR disk full\nINFO ok\nERROR timeout\nDEBUG x\n"},
         verify='[ "$(cat errors.txt)" = "$(printf "ERROR disk full\nERROR timeout")" ]'),

    dict(id="t04_unique", partial='test -f unique.txt', prompt="Write the sorted unique lines of names.txt to unique.txt.",
         setup={"names.txt": "zoe\nadam\nzoe\nbob\nadam\n"},
         verify='[ "$(cat unique.txt)" = "$(printf "adam\nbob\nzoe")" ]'),

    dict(id="t05_csv_sum", partial='test -f total.txt', prompt="sales.csv has a numeric second column. Write the sum of all values in column 2 to total.txt (integer).",
         setup={"sales.csv": "item,amount\npen,10\nbook,25\nlamp,15\n"},
         verify='[ "$(tr -d " \n" < total.txt)" = "50" ]'),

    dict(id="t06_json_extract", partial='test -f port.txt', prompt='Read config.json and write the value of the key "port" to port.txt.',
         setup={"config.json": '{"host": "localhost", "port": 8080, "debug": true}\n'},
         verify='[ "$(tr -d " \n" < port.txt)" = "8080" ]'),

    dict(id="t07_json_build", partial='test -f out.json', prompt="Create out.json containing the JSON object with key name equal to alice and key age equal to 30 (integer).",
         setup={}, verify='python3 -c "import json;d=json.load(open(\'out.json\'));assert d[\'name\']==\'alice\' and d[\'age\']==30"'),

    dict(id="t08_replace", partial='grep -q dog poem.txt', prompt="In the file poem.txt replace every occurrence of the word cat with dog and save the result back to poem.txt.",
         setup={"poem.txt": "the cat sat\na cat naps\nthe end\n"},
         verify='! grep -q cat poem.txt && [ "$(grep -c dog poem.txt)" = "2" ]'),

    dict(id="t09_concat", partial='test -f merged.txt', prompt="Merge a.txt and b.txt into merged.txt, a.txt first then b.txt.",
         setup={"a.txt": "one\ntwo\n", "b.txt": "three\n"},
         verify='[ "$(cat merged.txt)" = "$(printf "one\ntwo\nthree")" ]'),

    dict(id="t10_sha", partial='test -f hash.txt', prompt="Write the sha256 checksum of the file target.bin to hash.txt (just the hash, no filename).",
         setup={"target.bin": "hello world\n"},
         verify='[ "$(tr -d " \n" < hash.txt)" = "$(sha256sum target.bin | cut -d" " -f1)" ]'),

    dict(id="t11_wordfreq", partial='test -f top.txt', prompt="Write the single most frequent word in words.txt to top.txt (lowercase, no punctuation).",
         setup={"words.txt": "red blue red green red blue\n"},
         verify='[ "$(tr -d " \n" < top.txt)" = "red" ]'),

    dict(id="t12_dedupe", partial='test -f clean.txt', prompt="Remove consecutive duplicate lines from seq.txt preserving order, write to clean.txt.",
         setup={"seq.txt": "a\na\nb\nb\nb\nc\na\n"},
         verify='[ "$(cat clean.txt)" = "$(printf "a\nb\nc\na")" ]'),

    dict(id="t13_timestamps", partial='test -f times.txt', prompt="Extract every timestamp of the form HH:MM:SS from events.log into times.txt, one per line, preserving order.",
         setup={"events.log": "boot 09:15:01 ok\nerror 10:02:33 x\nshutdown 23:59:59\n"},
         verify='[ "$(cat times.txt)" = "$(printf "09:15:01\n10:02:33\n23:59:59")" ]'),

    dict(id="t14_rename", partial='test -f a.md', prompt="Rename every file ending in .txt in the current directory to end in .md instead.",
         setup={"a.txt": "1\n", "b.txt": "2\n", "c.md": "3\n"},
         verify='[ -f a.md ] && [ -f b.md ] && [ ! -f a.txt ] && [ ! -f b.txt ] && [ -f c.md ]'),

    dict(id="t15_count_matches", partial='test -f count.txt', prompt="Count how many times the word foo appears in text.txt and write the integer to count.txt.",
         setup={"text.txt": "foo bar foo\nfoo baz\nqux\n"},
         verify='[ "$(tr -d " \n" < count.txt)" = "3" ]'),

    dict(id="t16_permissions", partial='test -f run.sh', prompt="Make the file run.sh executable by its owner (add execute permission).",
         setup={"run.sh": "#!/bin/bash\necho hi\n"},
         verify='[ -x run.sh ]'),

    dict(id="t17_archive", partial='test -f bundle.tar.gz', prompt="Create a gzip-compressed tar archive named bundle.tar.gz containing the directory proj/ and its contents.",
         setup={"proj/x.txt": "x\n", "proj/y.txt": "y\n"},
         verify='tar tzf bundle.tar.gz | grep -q "proj/x.txt" && tar tzf bundle.tar.gz | grep -q "proj/y.txt"'),

    dict(id="t18_extract_archive", partial='test -f secret_copy.txt', prompt="Extract bundle.tar.gz then write the contents of proj/secret.txt to secret_copy.txt.",
         setup={}, verify='[ "$(cat secret_copy.txt)" = "s3cret" ]',
         pre="mkdir -p proj && printf 's3cret\\n' > proj/secret.txt && tar czf bundle.tar.gz proj && rm -rf proj"),

    dict(id="t19_base64", partial='test -f decoded.txt', prompt="The file blob.b64 contains base64 text. Decode it and write the decoded bytes to decoded.txt.",
         setup={"blob.b64": "aGVsbG8gcmwK\n"},
         verify='[ "$(cat decoded.txt)" = "hello rl" ]'),

    dict(id="t20_csv_filter", partial='test -f passed.csv', prompt="From data.csv keep only rows whose score column (third column) is greater than 50, including the header, and write them to passed.csv.",
         setup={"data.csv": "name,id,score\na,1,80\nb,2,45\nc,3,60\n"},
         verify='[ "$(wc -l < passed.csv)" = "3" ] && grep -q "^a,1,80" passed.csv && grep -q "^c,3,60" passed.csv && ! grep -q "^b,2,45" passed.csv'),

    dict(id="t21_largest", partial='test -f largest.txt', prompt="Find the name of the largest file in the directory and write just its name to largest.txt.",
         setup={"small.txt": "x", "big.txt": "y" * 400, "mid.txt": "z" * 60},
         verify='[ "$(tr -d " \n" < largest.txt)" = "big.txt" ]'),

    dict(id="t22_reverse", partial='test -f reversed.txt', prompt="Write the lines of input.txt to reversed.txt in reverse order (last line first).",
         setup={"input.txt": "one\ntwo\nthree\n"},
         verify='[ "$(cat reversed.txt)" = "$(printf "three\ntwo\none")" ]'),

    dict(id="t23_replace_ext", partial='grep -q WARNING a.log', prompt="In all .log files replace the string WARN with WARNING in place.",
         setup={"a.log": "WARN low\nINFO ok\n", "b.log": "WARN again\n"},
         verify='! grep -q WARN a.log && ! grep -q WARN b.log && grep -q WARNING a.log && grep -q WARNING b.log'),

    dict(id="t24_sum_numbers", partial='test -f sum.txt', prompt="The file nums.txt has one integer per line. Write their sum to sum.txt.",
         setup={"nums.txt": "4\n8\n15\n16\n23\n42\n"},
         verify='[ "$(tr -d " \n" < sum.txt)" = "108" ]'),

    dict(id="t25_json_array", partial='test -f users.json', prompt='Create users.json containing a JSON array of two objects, each with keys "name" and "active" (boolean): alice active true, bob active false.',
         setup={}, verify='python3 -c "import json;d=json.load(open(\'users.json\'));assert isinstance(d,list) and len(d)==2 and d[0][\'name\']==\'alice\' and d[0][\'active\'] is True and d[1][\'name\']==\'bob\' and d[1][\'active\'] is False"'),
]

SYSTEM = """You are an AI agent controlling a bash terminal. You are given a task and the contents of the working directory.
Respond with ONLY a JSON object, no markdown fences, in exactly this schema:
{"analysis": "<what you observe>", "plan": "<what you will do next>", "commands": [{"keystrokes": "<bash command>", "duration": 0.1}], "task_complete": false}
Rules:
- Issue one or a few shell commands per turn. Each command runs in the working directory.
- When you believe the task is done, set "task_complete": true and use an empty commands list.
- Never use interactive programs, sudo, or network access."""


def render(task, history, listing):
    parts = [f"TASK: {task['prompt']}", "", "WORKING DIRECTORY:", listing, ""]
    if history:
        parts.append("TRANSCRIPT:")
        for h in history:
            parts.append(f"$ {h['cmd']}")
            if h["out"].strip():
                parts.append(h["out"][:1200])
            if h["rc"] != 0:
                parts.append(f"[exit {h['rc']}]")
        parts.append("")
    parts.append("Respond with the JSON object now.")
    return "\n".join(parts)
