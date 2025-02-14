FROM pytorch/pytorch
RUN apt-get install python-3.11.6 \
COPY . .

