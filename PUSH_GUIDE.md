# GitHub에 push하는 방법

이 파일은 `surface/` 디렉토리를 받으신 후, `github.com/leonyoon-3dai/surface`로 push하는 방법입니다. push가 끝나면 이 파일은 지우셔도 됩니다.

## 방법 1: GitHub CLI (`gh`) 사용 — 가장 간단

`gh`가 설치되어 있고 인증이 되어 있다면:

```bash
cd surface
git init
git add .
git commit -m "Initial commit: neural surface reconstruction notes"
git branch -M main
gh repo create leonyoon-3dai/surface --public --source=. --push
```

끝입니다. `--private`으로 바꾸면 private repo가 됩니다.

## 방법 2: GitHub 웹에서 먼저 repo 만들기

1. https://github.com/new 에서 repo 이름 `surface`로 생성 (README/gitignore/LICENSE는 **체크하지 마세요** — 이미 파일에 포함되어 있습니다)
2. 로컬에서:

```bash
cd surface
git init
git add .
git commit -m "Initial commit: neural surface reconstruction notes"
git branch -M main
git remote add origin https://github.com/leonyoon-3dai/surface.git
git push -u origin main
```

## 확인 포인트

push 후 GitHub 페이지에서:
- README의 LaTeX 수식이 렌더링되는지 (GitHub는 `$...$`, `$$...$$` 지원)
- `figures/` 폴더의 SVG 이미지가 README에 정상 표시되는지

수식이 렌더링되지 않으면 GitHub 계정의 Markdown 설정을 확인하시거나, `$$...$$` 블록 주변에 빈 줄이 있는지 확인하세요.

## LICENSE 수정

`LICENSE` 파일의 저작권자 이름(`leonyoon-3dai`)을 본인 이름으로 바꾸고 싶으시면 편집 후 commit하시면 됩니다.
