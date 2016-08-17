sudo docker rm -f SPARK2
sudo docker run -ti -v /vagrant/apps/Kaggle-Walmart-fcst-in-Spark:/myapp --name SPARK2 spark2 /bin/bash