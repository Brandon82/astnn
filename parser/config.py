# The keys for the dictionary of the dataset
program_key = 'https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/program'
difference_key = 'https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/difference'
equivalence_key = 'https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/equivalence'
operator_key = 'https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/operator'
mutant_key = "https://b2share.eudat.eu/records/153db16ce2f6401298a9aea8b0ab9781/Mutant"
keys_to_ignore = ("http://schema.org/name", "http://schema.org/person", "http://schema.org/familyName", "http://schema.org/URL", "http://schema.org/person")



original_program_path = '../dataset/original_programs/'
dataset_path = '../dataset/'
dataset_file = '../dataset/dataset.ttl'
mutant_save_path = '../dataset/mutated_data/'
mutant_save_split_path = '../dataset/mutated_data_split/'
save_mutants_to_file = True
save_only_methods = True
num_mutants_to_parse = 0    # 0 = parse all mutants