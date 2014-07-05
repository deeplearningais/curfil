#!/usr/bin/perl -w

#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################

use File::Find;
use Carp::Assert;
use Cwd;


sub get_license{
    $is_python = shift;
    open LIC, "<$basedir/LICENSE.txt" or die $!;
    my @lines;
    while(<LIC>){
        next if $is_python and /^#if 0$/;
        next if $is_python and /^#endif$/;
        push @lines, $_;
    }
    return join "", @lines;
}

sub add_license{
    $file = shift;
    open FH, "<$file" or die $!;
    open OUT, ">$file.new" or die $!;

    $is_python = $file =~ /\.py$/;

    $firstline = "";
    if($is_python){
        $firstline = <FH>;
        if($firstline =~ /python/){
            print OUT $firstline;
            $firstline = "";
        }
    }
    print OUT get_license($is_python);
    print OUT $firstline;
    while(<FH>){
        print OUT $_;
    }
    close OUT;
    close FH;
    $ret = system("mv '$file.new' '$file'");
    assert($ret == 0);
}

sub has_license{
    $filename = shift;
    return 0 == system("grep -q Copyright $filename");
}

sub wanted{
    $filename = $_;
    return if $filename !~ /\.(py|h|hpp|cuh|c|cpp|cu)$/;
    #`git checkout $filename`;
    return if has_license($filename);
    $cwd = cwd();
    print "Adding license to $filename ($cwd)\n";
    add_license($filename);
    `git add $filename`;
}

$basedir = cwd();
sub main{
    find(\&wanted, $basedir);
}

main()
