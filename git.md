**！！！ 文件暂存区不影响任何操作，千万不要取消暂存**
## 本地仓库
- ```git init:``` 初始化仓库
- ```git status -s:```  查看文件变动状态 
  - **红色**表示只修改，还未进入暂存区
  - **绿色**表示文件进入暂存区内
- ```git check-ignore -v filename:```  查看指定文件是否被忽略 
  - 忽略则输出：.gitignore: 行号:*.txt      filename
  - 未忽略则无任何输出
- ```git add filename:``` 将filename添加到暂存（无返回）
- ```git add .:``` 将所有**修改、新增、删除**等变化内容的文件添加到暂存区
- ```git commit -m "[message]":``` 将暂存区文件进行**提交**到本地仓库
  - message表示本地修改的注释

## 远程仓库
- ```git remote add alias url:``` 连接远程仓库
  - **alias** 表示连接名称
  - **url** 表示远程仓库地址   
- ```git remote -v:``` 查看连接远程仓库的信息
- ```git remote set-url alias new-url:``` 修改远程仓库地址
- ```git push alias branch:``` 将本地git仓库提交内容推送到远程仓库的具体分支中
- ```git pull alias branch:``` 获取远程仓库的分支内容，自动与本地内容合并
- ```git fetch alias:``` 获取远程仓库的更新,不合并本地内容
- ```git branch:``` 查看当前本地分支
- ```git merge know/main:``` 从远程仓库know中获取main分支内容合并到本地
  - 添加```--allow-unrelated-histories``` 表示两个无共同提交历史的独立项目强行合并
- ```git log:``` 查看提交历史
  - 添加 ```--oneline``` 让输出更简洁
- ```git revert HEAD:``` 撤回一次本地提交（不影响远程内容）
  - 可以使用 ```git push know main``` 进行提交一次远程仓库（把本地撤回来的代码更新到远程仓库，实现覆盖错误提交）
  - 如果是首次合并的提交，需要添加```-m 1```进行最后一次父节点强制撤回
