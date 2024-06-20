# Version update pipeline

## Release conventions

See full [development roadmap](./roadmap.md) for past and planned features corresponding to each versions.
We use the following denomination for each version update (_a.k.a_ releases) :

- patch : from `*.*.{i}` to `*.*.{i+1}` $\Rightarrow$ minor modifications, bugfixes, code reformating, additional aliases for generators 
- minor : from `*.{i}.*` to `*.{i+1}.0` $\Rightarrow$ addition of new generators, new utility functions, new scripts, ...
- major : from `{i}.*.*` to `{i+1}.0.0` $\Rightarrow$ major changes in code structure, design and API

Here are some generic recommendation on release-triggering events :

1. patch version should be released every three months in case some only patch-type commits have been done
2. minor version should be released after merging a PR including new features (requires a version dump commit, see below ...)
3. major version are released when important changes have been done on a development branch named `v{i+1}-dev` hosted on the main repo. Requires a full update of the documentation and code, eventually some migration guide, etc ... Before merging `v{i+1}-dev` into `main`, a `v{i}-lts` branch is created from it to keep track of the old version, and eventually update it with some minor or patch releases until a pre-defined deprecation date defined in `docs/SECURITY.md`.

## Pipeline description

To release a new version, one need maintainer access to the `qmat` Github project, and execute the following steps :

1. Modify the version number in [`pyproject.toml`](https://github.com/Parallel-in-Time/qmat/blob/main/pyproject.toml)
2. Modify the version number and the release date in [`CITATION.cff`](https://github.com/Parallel-in-Time/qmat/blob/main/CITATION.cff)
3. (Minor & major update) update [`roadmap.md`](https://github.com/Parallel-in-Time/qmat/blob/main/docs/devdoc/roadmap.md) if not done already
4. (Major update) update [SECURITY.md](https://github.com/Parallel-in-Time/qmat/blob/main/docs/SECURITY.md) if not done already
5. Commit with message `XX: dump version` where `XX` are your initials
6. Manually run the ["Publish to PyPI 📦"](https://github.com/Parallel-in-Time/qmat/actions/workflows/publish.yml) workflow
7. [Draft a new release](https://github.com/Parallel-in-Time/qmat/releases/new) associated to a new tag `v*.*.*` (with `*.*.*` the new version, and the `+ Create new tag: ... on publish` button)
8. Find a cool title for the release, and describe what is new or changed (don't forget to thanks the non-maintainers authors)

And finally, click on `Publish release` 🚀
