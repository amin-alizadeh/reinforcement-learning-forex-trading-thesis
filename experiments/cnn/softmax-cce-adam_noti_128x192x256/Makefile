DIR ?= ../../../data/daily/
NN ?= 128,192,256
OUTPUT ?=
MODEL ?=

CUR := $(shell pwd)

ifeq ($(strip $(MODEL)),)
MODEL := $(CUR)/$(OUTPUT)/training/models
else
MODEL := $(CUR)/$(MODEL)
endif

OUTPUT_DIR := $(CUR)/$(OUTPUT)
ifeq ($(strip $(OUTPUT)),)
OUTPUT_DIR := $(CUR)
endif

DATA_DIR := $(CUR)/$(DIR)

default: train

train:
	python $(CUR)/../../../main.py  -m train \
	-o $(OUTPUT_DIR)/training \
	-td $(DATA_DIR) -f "Date<20200100" -jc Date \
	-bs 32 -inv 20000 -e 1500 -ms 32 --random_memory_sampling false  \
	-ps Close -cm 0.1 -nn $(NN) -cnn -cpu 1 -gpu 0 -ed 0.99999 -em 0.1 -eps 1.0 \
	-estart 1.0 -eend 0.1 -esteps 1400 \
	--activation "softmax" --loss "categorical_crossentropy" --optimizer "adam"

run:
	python $(CUR)/../../../main.py -m test \
	-o  $(OUTPUT_DIR)/test \
	-td $(DATA_DIR) -f "Date>20200100" -jc Date \
	-bs 32 -inv 20000 -e 1 -ms 32 --random_memory_sampling false  \
	-ps Close -cm 0.1 -nn $(NN) -cnn -cpu 1 -gpu 0 -em 0.05 -eps 0 -estart 0 -eend 0 \
	--activation "softmax" --loss "categorical_crossentropy" --optimizer "adam" \
	-lm $(MODEL)

best:
	python $(CUR)/../../../main.py -m test \
	-o  $(OUTPUT_DIR)/best \
	-td $(DATA_DIR) -f "Date>20200100" -jc Date \
	-bs 32 -inv 20000 -e 1 -ms 32 --random_memory_sampling false  \
	-ps Close -cm 0.1 -nn $(NN) -cnn -cpu 1 -gpu 0 -em 0.05 -eps 0 -estart 0 -eend 0 \
	--activation "softmax" --loss "categorical_crossentropy" --optimizer "adam" \
	-lm $(MODEL)/best