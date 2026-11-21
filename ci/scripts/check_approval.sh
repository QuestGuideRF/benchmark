#!/usr/bin/env bash
function LOG {
  echo [$0:${BASH_LINENO[0]}] $* >&2
}
LOG "[ИНФО] Начать проверку утверждения..."
EXIT_CODE=0
BENCHMARK_ROOT=$(cd $(dirname $0)/../../ && pwd)
declare -A FILE_APPROVAL_USER_MAP
FILE_APPROVAL_USER_MAP=(
  ["api/common/special_op_list.py"]="ДжеймсЛим-си ЗзШон Чжантинг2020"
)
LOG "[ИНФО] Получить список утверждений..."
declare -A APPROVALED_USER_MAP
approval_line=$(curl -H "Авторизация: токен ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/benchmark/pulls/${AGILE_PULL_ID}/reviews?per_page=10000 2> /dev/null)
[ $? -ne 0 ] && LOG "[FATAL] Получить информацию об обзоре с github faied" && exit -1
for user in $(echo $approval_line | jq .[].user.login | sed 'с|"||г')
do
  APPROVALED_USER_MAP[$user]="approvaled"
  LOG "[INFO] ${user} одобряет этот запрос."
done
for file in $(git diff --name-only upstream/master)
do
  if [ -n "${FILE_APPROVAL_USER_MAP[$file]}" ]
  then
    need_approval=1
    approval_user=""
    for user in ${FILE_APPROVAL_USER_MAP[$file]}
    do
      [ -n "${APPROVALED_USER_MAP[$user]}" ] && need_approval=0 && break
      approval_user="$user, ${approval_user}"
    done
    if [ $need_approval -ne 0 ]
    then
      EXIT_CODE=6
      LOG "[FATAL] You must have one RD (${approval_user%,*}) approval for the ${file}"
    fi
  fi
done
[ $EXIT_CODE -ne 0 ] && LOG "[FATAL] Проверка не одобрена." || LOG "[ИНФО] Проверьте успешность утверждения."
exit $EXIT_CODE