# cmeee
## .gitignore

忽略某些文件不上传

```
touch  .gitignore     #创建gitignore  
vim    .gitignore     #编辑文件，加入指定文件
```



```
*.a       # 忽略所有 .a 结尾的文件
!lib.a    # 但 lib.a 除外
/TODO     # 仅仅忽略项目根目录下的 TODO 文件，不包括 subdir/TODO
build/    # 忽略 build/ 目录下的所有文件
doc/*.txt # 会忽略 doc/notes.txt 但不包括 doc/server/arch.txt
```

本地缓存删除（改变成未track状态）

```
git rm -r --cached 文件/文件夹名字
git rm -r --cached .
```

## 如何从本地上传一个project

Initialize the local directory as a Git repository.

```
git init
```

Add the files in your new local repository. This stages them for the first commit.

```
git add .
```

or:

```
git add --all
```

Commit the files that you've staged in your local repository.

```
git commit -m 'upload'
```

Copy remote repository URL field from your GitHub repository, in the right sidebar, copy the remote repository URL.

In Terminal, add the URL for the remote repository where your local repostory will be pushed.

```
git remote add origin <remote repository URL>
```

git remote add origin https://github.com/Z-MU-Z/Image_captioning.git

 git remote add origin https://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@github.com/Z-MU-Z/cmeee.git

如果想要删除

git remote rm origin

Sets the new remote:

```
git remote -v
```

Push the changes in your local repository to GitHub.

```
git push origin master
```

结果竟然报错了

remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.

https://blog.csdn.net/qq_41646249/article/details/119777084 按这个方法申请token

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

 git remote set-url origin https://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@github.com/Z-MU-Z/cmeee.git

https://github.com/Z-MU-Z/cmeee.git

### …or create a new repository on the command line



```
echo "# cmeee" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Z-MU-Z/cmeee.git
git push -u origin main
```

### …or push an existing repository from the command line



```
git remote add origin https://github.com/Z-MU-Z/cmeee.git
git branch -M main
git push -u origin main
```

git config --global http.postBuffer 524288000



## 使用SSH

https://blog.csdn.net/weixin_42902669/article/details/114699679

```shell
git remote set-url origin git@github.com:Z-MU-Z/cmeee.git
```

## 版本回退

```
git log -5--pretty=oneline
```

 显示之前的版本

de17868d10aa54512fd6464ef2f7bba0802183c2 (HEAD -> master, origin/master) upload

```
git reset --hard HEAD^^
```

回到上上个版本 或者 git reset --hard HEAD~1

git reset --hard de17868 到对应版本



## 删除与回复

git checkout -- test.txt 从本地仓库捞回工作区。

git rm test.txt 从本地仓库也删除



## 分支操作

切换

```
git checkout name
```

查看

```
git branch
#同时查看远程的
git branch -a 
```

创建

```
git  branch name
```

合并 到master (必须切到master)

```
git merge name
```

删除本地

```
git branch -d name
```

删除远程

git push origin :remotebranch_name

push

git push origin name

拉取远程branch

```
git checkout -b localname origin/remote_name
```

## 分支冲突

git pull = git fetch + git merge



```
git pull origin master
git push origin master

```

git fetch
