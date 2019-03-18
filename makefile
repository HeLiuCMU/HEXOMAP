# Project management


# ----- targets ---- 
.PHONY: install test list clean

# run test on hexomap support modules
test:
	@echo "Running test on all supporting modules of hexomap"
	@nosetests -v hexomap/tests


# install dependencies
install:
	@pip install -r requirements.txt
	@echo "Remember to add hexomap folder to your python library path."


# list all possible target in this makefile
list:
	@echo "LIST OF TARGETS:"
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null \
	| awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
	| sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs


# clean all temp files
clean:
	@echo "Clean up workbench"
	rm  -fv   *.tmp
	rm  -fv   tmp_*