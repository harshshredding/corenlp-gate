# CLaC Stanford Parser

A gate plugin to parse using the latest stanford parsers. This plugin includes the following processing resources:
 - New Stanford Parser : Takes raw text as input and parses every sentence, outputing Token, Sentence, Dependency, and SyntaxTreeNode annotations.
 - Dependency Tree Viewer 1 : A dependency parse visualizer that assumes the "New Stanford Parser" has already been run. This visualizer makes use of the dependency info included in the token features by the "New Stanford Parser".
 - Dependency Node Generator : Assumes the "New Stanford Parser" has been run. It generates DependencyTreeNode annotations.
 - Dependency Tree Viewer 2 : An alternate dependency parse visualizer that assumes "Dependency Node Generator" has been run. This visualizer sorts tree nodes according to the sequence of words in the sentence, potentially making long parses more readable. 

## Installation
Clone repo

```sh
git clone http://vp/root/clac-stanford-parser.git
```
Install latest Maven from : https://maven.apache.org/install.html
Make sure you are using Java 1.8.
To check your Maven and Java installation do:
```sh
java -version 
openjdk version "1.8.0_312"
OpenJDK Runtime Environment (build 1.8.0_312-8u312-b07-0ubuntu1~18.04-b07)
OpenJDK 64-Bit Server VM (build 25.312-b07, mixed mode)

mvn --version 
Apache Maven 3.8.5 (3599d3414f046de2324203b78ddcf9b5e4388aa0)
Maven home: /home/claclab/Downloads/apache-maven-3.8.5
Java version: 1.8.0_312, vendor: Private Build, runtime: /usr/lib/jvm/java-8-openjdk-amd64/jre
Default locale: en_CA, platform encoding: UTF-8
OS name: "linux", version: "5.4.0-96-generic", arch: "amd64", family: "unix"

```
In the project directory : 

```sh
mvn install
```
This will install the plugin in the local maven repository which is located in ~/.m2.

Finally, load the plugin in Gate Developer register the plugin using the info in `~/pom.xml` . For eg :
 - groupID : ca.concordia.gate
 - artifactID: stanford-parser-corenlp
 - version: 1.0-SNAPSHOT