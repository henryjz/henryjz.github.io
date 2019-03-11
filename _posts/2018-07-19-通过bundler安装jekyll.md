---
layout: post
comments: true
categories: notes
---

- **新建一个空目录，作为blog的根目录。**

```
mkdir blog
cd ./blog
```

- **创建Gemfile，内容简单复制以下代码就行,命名为Gemfile**

```
source  'https://rubygems.org'
gem 'github-pages',group: :jekyll_plugins
```

- **安装ruby和ruby-dev**

```
sudo apt install ruby ruby-dev
```

- **通过ruby的安装器gem安装bundler**

```
sudo gem install bundler
```

- **通过bundler批量安装jekyll组件**

```
bundle install
```

到nokogiri时终止了安装，搜索网上的问题解决办法后，发现是一些第三方依赖库没有安装，linux下在终端中输入以下命令，在手动安装nokogiri,成功！

```
sudo apt-get install libz-dev libiconv-hook1 libiconv-hook-dev
```

```
sudo gem install nokogiri -v '1.8.4'
```

- **回过头来根据Gemfile安装jekyll**

```
bundle install
```

- 试试能不能启动jekyll,成功！在浏览器中网址栏输入localhost:4000,验证jekyll的运行情况

```
bundle exec jekyll serve
```