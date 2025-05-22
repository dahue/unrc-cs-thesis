### Answer the question by SQLite SQL query only and with no explanation. You must minimize SQL execution time while ensuring correctness.
### Sqlite SQL tables, with their properties:
#
{schema_ddl}
#
### Here is some data information about database references.
#
{data_samples}
#
### Foreign key information of SQLite tables, used for table joins:
#
{foreign_keys}
#
### Some example pairs of questions and corresponding SQL queries are provided based on similar questions:
### How many farms are there?
SELECT count(*) FROM farm
### What is the average, minimum, and maximum age for all French singers?
SELECT avg(age) ,min(age) ,max(age) FROM singer WHERE country = 'France'
### Show the ID of the high schooler named Kyle.
SELECT ID FROM Highschooler WHERE name = 'Kyle'
### {question}
