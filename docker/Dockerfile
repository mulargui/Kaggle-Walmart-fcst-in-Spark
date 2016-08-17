FROM ubuntu:latest

# Install Java
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update && apt-get -qq -y install default-jre default-jdk
ENV JAVA_HOME /usr/lib/jvm/default-java

# Install maven
# RUN apt-get -qq install -y maven

# Install Scala
# RUN  apt-get -qq install -y wget 
# RUN wget -q http://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.deb
# RUN dpkg -i scala-2.11.8.deb
# ENV PATH $PATH:/usr/local/scala/bin	

# Install Python
# installed by default in ubuntu (python3)
ENV PYSPARK_PYTHON python3

# Install R
# RUN apt-get -qq install -y r-base

# Install Spark
RUN  apt-get -qq install -y curl 
RUN curl -s http://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz | tar -xz -C /usr/local/
RUN ln -s /usr/local/spark-2.0.0-bin-hadoop2.7 /usr/local/spark
RUN sed 's/INFO/ERROR/' /usr/local/spark/conf/log4j.properties.template > /usr/local/spark/conf/log4j.properties
RUN sed -i 's/WARN/ERROR/' /usr/local/spark/conf/log4j.properties
ENV SPARK_HOME /usr/local/spark
ENV PATH $PATH:/usr/local/spark/bin

EXPOSE 4040

# ENTRYPOINT ["/usr/local/spark/bin/spark-shell --master local[2]"]	
# ENTRYPOINT ["/usr/local/spark/bin/spark-submit --master local[2] program params"]	
