#!/usr/bin/python

import cgi, cgitb

# Create instance of FieldStorage
form = cgi.FieldStorage()

# Get data from fields
t_size = form.getvalue('t_size')

print "Content-type:text/html\r\n\r\n"
print "<html>"
print "<head>"
print "<title>Hello - Second CGI Program</title>"
print "</head>"
print "<body>"
print "<h2>%s</h2>" % (t_size)
print "</body>"
print "</html>"
