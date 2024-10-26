import yaml

with open('configs/languages.yaml', 'r', encoding='utf-8') as languages:
    languages = yaml.load(languages, Loader=yaml.FullLoader)

    main_title = languages['main-title']
    open_title = languages['open-title']
    save_title = languages['save-title']

    file_menu = languages['file-menu']
    help_menu = languages['help-menu']

    open_action = languages['open-action']
    save_action = languages['save-action']
    exit_action = languages['exit-action']

    types_description = languages['types-description']
