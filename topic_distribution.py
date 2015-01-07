 12 def dynamic_topic_model_import_data():
 13         global path_to_topics
 14         mallet_path = 'mallet_dir/mallet-2.0.7/bin/mallet'
 15         source_dir = path_to_topics + 'input_train/'
 16         des_dir = path_to_topics + 'output_train/'
 17         cmd = mallet_path + ' import-dir --input ' + source_dir + ' --output ' + des_dir +'one_train_output_data.mallet ' + '--remove-stopwords true --keep-sequence true --keep-sequence-bigrams true'
 18         r = envoy.run(cmd)
 19         print(r.status_code)
 20         return
 21 def dynamic_topic_model_import_data_for_inference():
 22         global path_to_topics
 23         data_dir = path_to_topics + 'output_train/one_train_output_data.mallet'
 24         source_dir = path_to_topics + 'input_test/'
 25         data_output_dir = path_to_topics + 'output_test/one_infer_output_data.mallet'
 26         mallet_path = 'mallet_dir/mallet-2.0.7/bin/mallet'
 27         cmd = mallet_path + ' import-dir --input ' + source_dir  + ' --output ' + data_output_dir +' --use-pipe-from ' + data_dir
 28         r = envoy.run(cmd)
 29         print(r.status_code)
 30         return
 31
 32 def dynamic_topic_model_train_data():
 33         global path_to_topics
 34         mallet_path = 'mallet_dir/mallet-2.0.7/bin/mallet'
 35         source_dir = path_to_topics + 'input_train/'
 36         data_output = path_to_topics + 'output_train/'
 37         test_output = path_to_topics + 'output_test/lda_inferencer.mallet'
 38         cmd = mallet_path + ' train-topics ' + '--input ' + data_output +'one_train_output_data.mallet --inferencer-filename ' + test_output + ' --num-topics 10 --output-doc-topics ' + data_output +'tra    in_topic_composition.txt'
 39         r = envoy.run(cmd)
 40         print(r.status_code)
 41         return
 42
 43 def dynamic_topic_model_for_inference():
 44         global path_topics
 45         mallet_path = 'mallet_dir/mallet-2.0.7/bin/mallet'
 46         data_file_path = path_to_topics + 'output_test/one_infer_output_data.mallet'
 47         cmd  = mallet_path + ' infer-topics ' + '--input ' + data_file_path + ' --inferencer ' + path_to_topics + 'output_test/lda_inferencer.mallet --output-doc-topics ' + path_to_topics+'output_test/t    est_topic_composition.txt'
 48         r = envoy.run(cmd)
 49         print(r.status_code)
 50         return
 51
 52 def remove_files():
 53         global path_to_topics
 54         rm_input_train = path_to_topics + 'input_train/*'
 55         rm_input_test = path_to_topics + 'input_test/*'
 56         rm_output_train = path_to_topics + 'output_train/*'
 57         rm_output_test = path_to_topics + 'ouput_test/*'
 58         r = envoy.run('sh -c "rm -rf  mallet_dir/input_train/*"')
 59         print(r.status_code)
 60         r = envoy.run('sh -c "rm -rf mallet_dir/input_test/*"')
 61         print(r.status_code)
 62         r = envoy.run('sh -c "rm -rf mallet_dir/output_train/*"')
 63         print(r.status_code)
 64         r = envoy.run('sh -c "rm -rf mallet_dir/output_test/*"')
 65         print(r.status_code)
 66         return
                        
