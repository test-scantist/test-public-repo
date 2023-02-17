/* You do not need to change anything up here. */
package lexer;

import frontend.Token;
import static frontend.Token.Type.*;

%%

%public
%final
%class Lexer
%function nextToken
%type Token
%unicode
%line
%column

%{
	/* These two methods are for the convenience of rules to create token objects.
	* If you do not want to use them, delete them
	* otherwise add the code in 
	*/
	
	private Token token(Token.Type type) {
		return token(type, yytext());
	}
	
	/* Use this method for rules where you need to process yytext() to get the lexeme of the token.
	 *
	 * Useful for string literals; e.g., the quotes around the literal are part of yytext(),
	 *       but they should not be part of the lexeme. 
	*/
	private Token token(Token.Type type, String text) {
		return new Token(type, yyline, yycolumn, text);
	}
%}

/* This definition may come in handy. If you wish, you can add more definitions here. */
WhiteSpace = [ ] | \t | \f | \n | \r

/* Identifier */
Identifier = [A-Za-z][A-Za-z0-9_]*

/* Keyword */
Boolean = "boolean"
Break = "break"
Else = "else"
False = "false"
If = "if"
Import = "import"
Int = "int"
Module = "module"
Public = "public"
Return = "return"
True = "true"
Type = "type"
Void = "void"
While = "while"

/* Literal */
Int_literal = [0-9]+
// String_literal = \"(\\.|[^"\\])*\"
String_literal = \"[^\"\n]*\"

%%
/* put in your rules here.    */
// punctuation symbols
","		{ return token(COMMA); }						/* , */
"["		{ return token(LBRACKET); }						/* [ */
"{"		{ return token(LCURLY); }						/* { */
"("		{ return token(LPAREN); }						/* ( */
"]"		{ return token(RBRACKET); }						/* ] */
"}"		{ return token(RCURLY); }						/* } */
")"		{ return token(RPAREN); }						/* ) */
";"		{ return token(SEMICOLON); }					/* ; */

// operators
"/"		{ return token(DIV); }						/* / */
"=="	{ return token(EQEQ); }						/* == */
"="		{ return token(EQL); }						/* = */
">="	{ return token(GEQ); }						/* >= */
">"		{ return token(GT); }						/* > */
"<="	{ return token(LEQ); }						/* <= */
"<"		{ return token(LT); }						/* < */
"-"		{ return token(MINUS); }					/* - */
"!="	{ return token(NEQ); }						/* != */
"+"		{ return token(PLUS); }						/* + */
"*"		{ return token(TIMES); }					/* * */

{Boolean} { return token(BOOLEAN); }
{Break} { return token(BREAK); }
{Else} { return token(ELSE); }
{False} { return token(FALSE); }
{If} { return token(IF); }
{Import} { return token(IMPORT); }
{Int} { return token(INT); }
{Module} { return token(MODULE); }
{Public} { return token(PUBLIC); }
{Return} { return token(RETURN); }
{True} { return token(TRUE); }
{Type} { return token(TYPE); }
{Void} { return token(VOID); }
{While} { return token(WHILE); }

{Identifier} { return token(ID); }

{Int_literal} { return token(INT_LITERAL); }
{String_literal} { return token(STRING_LITERAL, yytext().substring(1, yylength() - 1)); }

{WhiteSpace} {}

/* You don't need to change anything below this line. */
.							{ throw new Error("unexpected character '" + yytext() + "'"); }
<<EOF>>						{ return token(EOF); }
