#! usr/bin/env python
#! -*- coding: utf-8 -*-
import itertools
def get_efficient_combinations(strVal,strLen):
	list_of_combinations = []
	if(len(strVal)%2 != 0 ):
		str_op = strVal[1:]
	else:
	 	str_op = strVal[:]
	final_list_elem = []
	if(len(str_op)% 2 != 0):
		print('missing option value enter the option value and retry\n')
		sys.exit(0)
	#print('string lenght:' + str(str_op))
        for i in xrange(0,len(str_op),2):
		if(int(str_op[i+1].strip()) == 1):
			final_list_elem.append(str_op[i])
	print('final list: ' + str(final_list_elem))
	if(len(str_op) == 0 ):
		list_of_combinations.append([])
	for i in xrange(1,strLen + 1):
		temp_list = list(itertools.combinations(final_list_elem,i))
		list_of_combinations.extend(temp_list)
	return list_of_combinations, final_list_elem
def validation_check(list_of,global_feature_dict):
	for key,value in global_feature_dict.iteritems():
		global_feature_dict[key] = 1
	for each in list_of:
		if(global_feature_dict.get(each,None) == None):
			return True
	return False
def work_with_combinations():
	list_elem,final_list = get_efficient_combinations(['sab','1','show','1','dal','1','bha1','1','know','1'],10)
	count = 0
	new_list = list_elem[:]
	for each in new_list:
		print(each)
		for sub_each in each:
			print(sub_each.strip('\n')+'_train'+' ').strip('\n')
		count += 1
		#print('\n')
	print(str(count))
	return
#work_with_combinations()

