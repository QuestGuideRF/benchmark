import os
import argparse
import subprocess
from git import Repo
import requests
github_base_url ="https://api.github.com/repos/PaddlePaddle/Paddle/"
parser =argparse .ArgumentParser (description =__doc__ )
parser .add_argument (
"--repo_path",
type =str ,
default ='.',
help ="путь к репозиторию git")
parser .add_argument (
"--branch",
type =str ,
default ='',
help ="название ветки репо кассы")
parser .add_argument (
"--pr",
type =str ,
default ='',
help ="идентификатор запроса на извлечение")
parser .add_argument (
"--commit",
type =str ,
default ='',
help ="идентификатор фиксации")
parser .add_argument (
"--merge_before",
type =str ,
default ='false',
help ="извлечь код перед слиянием, значение bool, поддержка true и false")
def _query_commit_info_from_github (commit_id ):
    url =github_base_url +"commits/"+str (commit_id )
    try :
        response =requests .get (url )
        parent_commit_id =response .json ()["parents"][0 ]["sha"]
        print ("Родителем коммита %s является коммит %s"%(commit_id ,parent_commit_id ))
        result ={
        "parent_commit_id":parent_commit_id
        }
        return result
    except Exception as e :
        raise Exception ("запросить информацию о фиксации из github %s, ошибка: %s"%(
        url ,str (e )
        ))
def _query_pull_request_info_from_github (pr_id ):
    url =github_base_url +"pulls/"+str (pr_id )
    try :
        response =requests .get (url )
        branch_name =response .json ()["base"]["ref"]
        print ("pr %s ветка слияния: %s"%(pr_id ,branch_name ))
        result ={
        "branch_name":branch_name
        }
        return result
    except Exception as e :
        raise Exception ("запросить информацию о PR из github %s, ошибка: %s"%(
        url ,str (e )
        ))
def _parameters_check (args ):
    print ("путь_репо: %s"%args .repo_path )
    print ("филиал: %s"%args .branch )
    print ("пр: %s"%args .pr )
    print ("зафиксировать: %s"%args .commit )
    print ("merge_before: %s"%args .merge_before )
    repo_path =os .path .abspath (args .repo_path )
    branch =args .branch
    pr =args .pr
    commit_id =args .commit
    merge_before =args .merge_before
    if merge_before .strip ()not in ["true","false"]:
        raise Exception ("merge_before %s должно иметь значение true или false"%str (repo_path ))
    if not os .path .exists (repo_path ):
        raise Exception ("repo_path %s not exists"%str (repo_path ))
    repo =Repo (repo_path )
    if repo .bare :
        raise Exception ("repo_path %s пуст"%str (repo_path ))
    if not branch and not pr and not commit_id :
        raise Exception ("--branch, --pr, --commit требуется хотя бы один")
    if pr and commit_id :
        raise Exception ("--pr, --commit — конфликт, можно просто установить один")
    if merge_before .strip ()=="true"and not pr and not commit_id :
        raise Exception ("когда --merge_before имеет значение true, --pr или --commit требуется хотя бы один")
def _process_git_repo (args ):
    _parameters_check (args )
    repo_path =os .path .abspath (args .repo_path )
    branch =args .branch
    pr =args .pr
    commit_id =args .commit
    merge_before =args .merge_before
    if merge_before .strip ()=="true":
        merge_before =True
    else :
        merge_before =False
    if not merge_before :
        if branch and branch !="develop"and not pr :
            commands ="cd %s && git checkout -b %s origin/%s"%(
            repo_path ,branch ,branch
            )
            print (commands )
            status ,output =subprocess .getstatusoutput (commands )
            print (output )
            if status !=0 :
                raise Exception ("Ошибка ветки git checkout %s: %s"%(branch ,output ))
        if pr :
            commands ="cd %s && git fetch origin pull/%s/head:%s"%(
            repo_path ,str (pr ),"пр-%s"%str (pr )
            )
            print (commands )
            status ,output =subprocess .getstatusoutput (commands )
            print (output )
            if status !=0 :
                raise Exception ("Ошибка git fetch pr %s: %s"%(str (pr ),output ))
            pr_info =_query_pull_request_info_from_github (pr )
            pr_into_branch =pr_info ["branch_name"]
            commands ="cd %s && git checkout -b merge-pr-%s origin/%s"%(
            repo_path ,pr ,pr_into_branch
            )
            print (commands )
            status ,output =subprocess .getstatusoutput (commands )
            print (output )
            if status !=0 :
                raise Exception ("Ошибка ветки git checkout %s: %s"%(pr_into_branch ,output ))
            commands ="cd %s && git merge %s"%(
            repo_path ,"пр-%s"%str (pr )
            )
            print (commands )
            status ,output =subprocess .getstatusoutput (commands )
            print (output )
            if status !=0 :
                raise Exception ("Ошибка git merge pr %s: %s"%(str (pr ),output ))
            else :
                return 0
        if commit_id :
            commands ="cd %s && git checkout -b %s %s"%(
            repo_path ,"совершить-%s"%commit_id ,commit_id
            )
            print (commands )
            status ,output =subprocess .getstatusoutput (commands )
            print (output )
            if status !=0 :
                raise Exception ("Ошибка фиксации идентификатора git checkout %s: %s"%(str (pr ),output ))
            else :
                return 0
    else :
        if pr :
            pr_info =_query_pull_request_info_from_github (pr )
            pr_into_branch =pr_info ["branch_name"]
            if pr_into_branch !="develop":
                commands ="cd %s && git checkout -b %s origin/%s"%(
                repo_path ,pr_into_branch ,branch
                )
                print (commands )
                status ,output =subprocess .getstatusoutput (commands )
                print (output )
                if status !=0 :
                    raise Exception ("Ошибка ветки git checkout %s: %s"%(pr_into_branch ,output ))
                else :
                    return 0
            else :
                commands ="cd %s && git checkout development"%(
                repo_path
                )
                print (commands )
                status ,output =subprocess .getstatusoutput (commands )
                print (output )
                if status !=0 :
                    raise Exception ("Ошибка ветки git checkout %s: %s"%(pr_into_branch ,output ))
                else :
                    return 0
        if commit_id :
            commit_info =_query_commit_info_from_github (commit_id )
            parent_commit_id =commit_info ["parent_commit_id"]
            commands ="cd %s && git checkout -b %s %s"%(
            repo_path ,"совершить-%s"%parent_commit_id ,parent_commit_id
            )
            print (commands )
            status ,output =subprocess .getstatusoutput (commands )
            print (output )
            if status !=0 :
                raise Exception ("Ошибка фиксации идентификатора git checkout %s: %s"%(str (pr ),output ))
            else :
                return 0
if __name__ =='__main__':
    args =parser .parse_args ()
    _process_git_repo (args )