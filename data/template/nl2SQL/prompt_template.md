[
{
"name": "nl2SQL_prompt_template_00",
"description": "",
"prompt_template": """
[QUESTION]{question}[/QUESTION]
[SCHEMA]{schema_ddl}[/SCHEMA]
[SQL]
"""
},
{
"name": "nl2SQL_prompt_template_11",
"description": "",
"prompt_template": """
[QUESTION]{question}[/QUESTION]
[SCHEMA]{schema_ddl}[/SCHEMA]
[FK]{foreign_keys}[/FK]
[SQL]
"""
}
]