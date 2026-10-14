# PCG shared-memory opt-in capacity — successor observations

These separate plans use internal source `a17e3e631e52e927bb6b81675b9a8f958d473d3d` and its matching resource-only probe. RTX gets N256; Jetson gets N256 and conditional N512. The prior unchanged-source capacity plans/results and the 293 base task plans retain their pins. GTX1070's 48 KiB block limit cannot support these requests; RTX N512 exceeds its 99 KiB opt-in block limit, so neither gets a new task grid for those sizes.

Felix supplied current unchanged-source N256 resource rejections on RTX and Jetson: 76,484 dynamic+32 static bytes exceeded the configured 49,152 allowance; the attempted launch returned `invalid argument`. Those completed resource measurements remain separate (source fa6, runner 7572137; archives not imported here). **Repeat N256 capacity after opt-in** with the successor below. Jetson N512 is a new observation, not a repeat of a measured result.

The successor configures the actual PCG kernel only when its request exceeds the default allowance and the device/kernel can support it. The probe records default/opt-in/effective limits, compiled static memory, thread limits, whether opt-in was applied and configuration errors. It launches **zero kernels** and never passes null operands to a valid kernel. A fit is a resource candidate; the separate finite single-solve readiness is still required before any task pilot. Valid unsupported-capacity outcomes are completed measurements; CUDA query/set/verification failures are execution errors even if the native probe returns exit 0. Raw JSON, stderr and return codes are retained.

After publication, in the existing target CUDA environment:

```bash
git pull --ff-only origin experiment/simple-runner
OPTIN_PYTHON="$(python3 -c 'import sys; print(sys.executable)')"
OPTIN_MACHINE=jetson # alternatively rtx5060
OPTIN_PLAN="experiments/launch/task_parameter_capacity_optin/$OPTIN_MACHINE.json"
OPTIN_OUTPUT="$PWD/.experiment-results/task-parameter-capacity-optin-20261014-$OPTIN_MACHINE"
OPTIN_CACHE="$PWD/.experiment-cache/task-parameter-capacity-optin-20261014-$OPTIN_MACHINE"
"$OPTIN_PYTHON" -c 'import json,sys; from pathlib import Path; sys.path.insert(0,"experiments"); from machine_launch import verify_sources; verify_sources(json.loads(Path(sys.argv[1]).read_text()))' "$OPTIN_PLAN"
"$OPTIN_PYTHON" experiments/run.py --plan "$OPTIN_PLAN" --machine "$OPTIN_MACHINE" --python "$OPTIN_PYTHON" --output "$OPTIN_OUTPUT" --cache "$OPTIN_CACHE" --only capacity-N256 --dry-run
"$OPTIN_PYTHON" experiments/run.py --plan "$OPTIN_PLAN" --machine "$OPTIN_MACHINE" --python "$OPTIN_PYTHON" --output "$OPTIN_OUTPUT" --cache "$OPTIN_CACHE" --only capacity-N256
```

Review the resource JSON and then follow [native readiness and same-cell task pilot commands](../task_parameters_large/README.md). On Jetson, assess N256 resource fit, finite execution and pilot wall cost before repeating these two commands with `--only capacity-N512`. If N256 demonstrates a clear operational ceiling, preserve it and record N512 skipped with the reason. A bad reaching outcome alone is not a hardware ceiling. No adaptive orchestration or automatic retry is added.

Each capacity build compiles only its selected `resources_N` executable. It cannot block another horizon's build. Capacity builds and task/native-readiness builds are distinct: the latter contain N32 plus their one large N and qualify the actual study module. RTX live native readiness uses the live source/module, not the internal one; the shared resource-helper geometry alone is not whole-module qualification.

Runner-plan preparation used CPU checks and dry runs only. The source owner’s local GTX checks are recorded separately; supported-device opt-in, native readiness and task execution on RTX/Jetson remain pending. No remote publication is claimed. Publish the runner and exact source refs listed in the current main handoff, then use `verify_sources` to retrieve missing immutable refs. Future reusable adoption starts from the separately reviewed core commit; frozen historical timing and task results remain unchanged.
