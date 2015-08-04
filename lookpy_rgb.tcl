#!/bin/sh
# the next line restarts using wish \
\
# for UNIX: \
exec wish "$0" "$@" -colormap new

# for laptop:
# exec cygwish80 "$0" "$@" $1 $2

option add *background gray16

# for laptop:
# set tmppath C:/cygwin/tmp
# set callstring "cygwish80 -f C:/cygwin/home/cweber/lw/co02/look.tcl"

# for UNIX:
set UNAME [exec whoami]

# global variables can also be set after the proc's
set AUTOREFRESH 1
set ZOOM 2
set UNAME tmp

set tmppath $UNAME
set activations_side top    ;# set: top  or  left

# set wm title
set splittedname [split $UNAME /]            ;# split elems where "/"
set shortname [lindex $splittedname end]     ;# last element
wm title . $shortname

set callstring "wish -f /home/fias/cweber/bw/coco06/look.tcl $argv"

set AREA_LIST ""
set DISPLAYACTS 0
set DISPLAYWEIGHTS 0

if  {$argc == 0} {
    puts "./look_rgb.tcl a w 0 1 2 3 4"
    exit
}

foreach u $argv {
    if  {[string is integer $u]} {
        lappend AREA_LIST $u
    }
    if  {[string compare $u a] == 0} {
        set DISPLAYACTS 1
        puts "Displaying acts"
    }
    if  {[string compare $u w] == 0} {
        set DISPLAYWEIGHTS 1
        puts "Displaying weights"
    }
    if  {[string compare $u z1] == 0} {set ZOOM 1}
    if  {[string compare $u z2] == 0} {set ZOOM 2}
    if  {[string compare $u z3] == 0} {set ZOOM 3}
    if  {[string compare $u z4] == 0} {set ZOOM 4}
    if  {[string compare $u z5] == 0} {set ZOOM 5}
    if  {[string compare $u z6] == 0} {set ZOOM 6}    
}
if  {[expr $DISPLAYACTS + $DISPLAYWEIGHTS] == 0} {
    set DISPLAYACTS 1
    set DISPLAYWEIGHTS 1
    puts "Displaying acts and areas"
}
puts "AREA_LIST: $AREA_LIST"




# The next 3 proc's are for saving the window as gif image file
# http://mini.net/tcl/9127
# http://www.tek-tips.com/viewthread.cfm?qid=993058&page=1
proc captureWindow { win } {

   package require Img

   regexp {([0-9]*)x([0-9]*)\+([0-9]*)\+([0-9]*)} [winfo geometry $win] - w h x y

   # Make the base image based on the window
   set image [image create photo -format window -data $win]

   foreach child [winfo children $win] {
     captureWindowSub $child $image 0 0
   }

   return $image
}

proc captureWindowSub { win image px py } {

   if {![winfo ismapped $win]} {
     return
   }

   regexp {([0-9]*)x([0-9]*)\+([0-9]*)\+([0-9]*)} [winfo geometry $win] - w h x y

   incr px $x
   incr py $y

   # Make an image from this widget
   set tempImage [image create photo -format window -data $win]

   # Copy this image into place on the main image
   $image copy $tempImage -to $px $py
   image delete $tempImage

   foreach child [winfo children $win] {
     captureWindowSub $child $image $px $py
   }
}

proc windowToFile { win ct } {

   set image [captureWindow $win]

   set types {{"Image Files" {.gif}}}

   set filename [tk_getSaveFile -filetypes $types \
 	                        -initialfile capture$ct.gif \
                                -defaultextension .gif]

   if {[llength $filename]} {
       $image write -format gif $filename
       puts "Written to file: $filename"
   } else {
       puts "Write cancelled"
   }
   image delete $image
}




proc show_pic {fram pic name} {
    global firsttime
    global AUTOREFRESH
    global ZOOM
    global activations_side

    set catcherror [catch {set pic [image create photo $pic -palette 256 -file $pic]}]
    if  {$catcherror == 0} {
        image create photo copy$pic
        copy$pic copy $pic -zoom $ZOOM
    }

    if  {[string compare fifo [file type $pic]] == 0} {
        set AUTOREFRESH 1
        set highS 0
        set lowS  0
    } else {
        set fp [open $pic r]
        seek $fp 0
        gets $fp line
        gets $fp line
        set highS [lindex [split [lindex [split $line :] 1] " "] 1]
        set lowS [lindex [split $line :] 2]
        close $fp
    }

    # cut "obs_" away
    set splittedname [split $name _]             ;# split elems where "_"
    set shortname [lreplace $splittedname 0 0]   ;# repl elem 0,0 with nothing
    set joinname [join $shortname _]             ;# join elems with "_"


  if  {$catcherror == 0} {
    if  {$firsttime} {
        label $fram.pics$name -image copy$pic
        label $fram.text$name -fg white -text "$joinname: [format %6.2f $lowS] .. [format %6.2f $highS]"
        if  {[string compare [string index $name 0] W]} {
            pack $fram.pics$name -side $activations_side
	} else {
            pack $fram.pics$name
	}
        pack $fram.text$name
    } else {
        $fram.pics$name configure -image copy$pic
        $fram.text$name configure -fg white -text "$joinname: [format %6.2f $lowS] .. [format %6.2f $highS]"
    }

    if  {$lowS == 0} {
        if  {$highS == 0} {
            $fram.text$name configure -fg grey
        }
    }
  }
}


proc make_pics {} {
    global AREA_LIST
    global picnames
    global tmppath
    global DISPLAYACTS
    global DISPLAYWEIGHTS

    foreach ar $AREA_LIST {

        set filelistact($ar) \
           [list obs_A_$ar obs_B_$ar obs_C_$ar obs_D_$ar obs_E_$ar obs_F_$ar obs_G_$ar obs_H_$ar obs_I_$ar obs_J_$ar \
                 obs_K_$ar obs_L_$ar obs_M_$ar obs_N_$ar obs_O_$ar obs_P_$ar obs_Q_$ar obs_R_$ar obs_S_$ar obs_T_$ar \
                 obs_U_$ar obs_V_$ar obs_W_$ar obs_X_$ar obs_Y_$ar obs_Z_$ar]

        set filelistweight($ar) \
           [list obs_W_${ar}_0  obs_W_${ar}_1  obs_W_${ar}_2  obs_W_${ar}_3  obs_W_${ar}_4  obs_W_${ar}_5  obs_W_${ar}_6  obs_W_${ar}_7  obs_W_${ar}_8  obs_W_${ar}_9\
                 obs_W_${ar}_10 obs_W_${ar}_11 obs_W_${ar}_12 obs_W_${ar}_13 obs_W_${ar}_14 obs_W_${ar}_15 obs_W_${ar}_16 obs_W_${ar}_17 obs_W_${ar}_18 obs_W_${ar}_19\
                 obs_V_${ar}_0  obs_V_${ar}_1  obs_V_${ar}_2  obs_V_${ar}_3  obs_V_${ar}_4  obs_V_${ar}_5  obs_V_${ar}_6  obs_V_${ar}_7  obs_V_${ar}_8  obs_V_${ar}_9\
                 obs_V_${ar}_10 obs_V_${ar}_11 obs_V_${ar}_12 obs_V_${ar}_13 obs_V_${ar}_14 obs_V_${ar}_15 obs_V_${ar}_16 obs_V_${ar}_17 obs_V_${ar}_18 obs_V_${ar}_19]

                 # obs_w_${ar}_0  obs_w_${ar}_1  obs_w_${ar}_2  obs_w_${ar}_3  obs_w_${ar}_4  obs_w_${ar}_5  obs_w_${ar}_6  obs_w_${ar}_7  obs_w_${ar}_8  obs_w_${ar}_9\
                 # obs_w_${ar}_10 obs_w_${ar}_11 obs_w_${ar}_12 obs_w_${ar}_13 obs_w_${ar}_14 obs_w_${ar}_15 obs_w_${ar}_16 obs_w_${ar}_17 obs_w_${ar}_18 obs_w_${ar}_19\
                 # obs_v_${ar}_0  obs_v_${ar}_1  obs_v_${ar}_2  obs_v_${ar}_3  obs_v_${ar}_4  obs_v_${ar}_5  obs_v_${ar}_6  obs_v_${ar}_7  obs_v_${ar}_8  obs_v_${ar}_9\
                 # obs_v_${ar}_10 obs_v_${ar}_11 obs_v_${ar}_12 obs_v_${ar}_13 obs_v_${ar}_14 obs_v_${ar}_15 obs_v_${ar}_16 obs_v_${ar}_17 obs_v_${ar}_18 obs_v_${ar}_19\
                 # ]

        if {$DISPLAYACTS} {

            foreach u $filelistact($ar) {

                if  {[file exists $tmppath/$u.pnm]} {

                    show_pic .fr$ar $tmppath/${u}.pnm $u

                    set picnames($tmppath/${u}.pnm) \
                        [list .fr$ar [file mtime $tmppath/${u}.pnm]]
                }
                if  {[file exists $tmppath/$u.pgm]} {

                    show_pic .fr$ar $tmppath/${u}.pgm $u

                    set picnames($tmppath/${u}.pgm) \
                        [list .fr$ar [file mtime $tmppath/${u}.pgm]]
                }
            }
        }


        if {$DISPLAYWEIGHTS} {

            foreach u $filelistweight($ar) {
                if  {[file exists $tmppath/$u.pnm]} {

                    show_pic .fr$ar $tmppath/${u}.pnm $u

                    set picnames($tmppath/${u}.pnm) \
                        [list .fr$ar [file mtime $tmppath/${u}.pnm]]
                }
                if  {[file exists $tmppath/$u.pgm]} {

                    show_pic .fr$ar $tmppath/${u}.pgm $u

                    set picnames($tmppath/${u}.pgm) \
                        [list .fr$ar [file mtime $tmppath/${u}.pgm]]
                }
            }
        }
    }
}


proc make_one_hist {trunc} {
    global firsttime
    global tmppath

    set fp [open $tmppath/$trunc.dat r]
    seek $fp 0

    set maxvalue 0
    for {set anz 0} {![eof $fp]} {incr anz} {
        gets $fp value($anz)
        if {$value($anz) > $maxvalue} {set maxvalue $value($anz)}
    }
    incr anz -1
    # puts "histogram: anz=$anz maxvalue=$maxvalue"
    close $fp



    set boxwidth 3
    set textheight 10
    set canvasheight [expr 200 + $textheight]

    if {$firsttime} {
        canvas .right.$trunc -width [expr $boxwidth * $anz] \
                        -height $canvasheight \
                        -background white
        pack .right.$trunc
    } else {
        .right.$trunc delete all
    }

    for {set i 0} {$i < $anz} {incr i} {
        if {$value($i) >= 0} {
           set value($i) [expr int($value($i) / $maxvalue * $canvasheight)]
           .right.$trunc create rectangle \
                 [expr $boxwidth * $i] \
                 [expr $canvasheight - $textheight] \
                 [expr $boxwidth * $i+$boxwidth] \
                 [expr $canvasheight - $textheight - $value($i)] \
                 -fill black -outline ""
        }
    }

    for {set i 0} {$i < $anz} {incr i 10} {
        .right.$trunc create rectangle \
              [expr $boxwidth * $i] \
              [expr $canvasheight - $textheight + 2] \
              [expr $boxwidth * $i+$boxwidth] \
              [expr $canvasheight - $textheight] \
              -fill red -outline ""
    }

    for {set i -3} {$i <= 3} {incr i} {
    .right.$trunc create text [expr ($anz / 2 + 10 * $i) * $boxwidth] \
                   [expr $canvasheight - $textheight] \
                   -text $i -anchor n
    }
}


proc make_hist {} {
    global tmppath

    set filelisthist {obs_distr_0 obs_distr_1 obs_distr_2}
    foreach u $filelisthist {
        if  {[file exists $tmppath/$u.dat]} {
            make_one_hist $u
        }
    }
}

proc doforever {} {
    global picnames
    global UNAME

    # puts "wp "

    if  {[file exists /tmp/$UNAME/tcl.pipe]} {
        set fp [open /tmp/$UNAME/tcl.pipe w]
        puts $fp 9
        close $fp
    }

    # puts "cp rb "

    if  {[file exists /tmp/$UNAME/tcl.back]} {
        set fp [open /tmp/$UNAME/tcl.back r]
        gets $fp line
        close $fp
    }

    # puts "cb "

    set update 0
    # puts -nonewline stdout "."
    # flush stdout
    foreach u [array names picnames] {
        if {[lindex $picnames($u) 1] != [file mtime $u]} {
             set update 1
        }
    }
    if  {$update} {
        make_pics
        # make_hist
    }

    after 10 doforever  ;# number is in msec
}



# Top level frames

foreach ar $AREA_LIST {
    frame .fr$ar
    pack .fr$ar -side left -fill both
}

frame .right
pack .right -side left -fill both


# Bitmap images

set firsttime 1
make_pics
make_hist
set firsttime 0


bind . <Button-1> {make_pics; make_hist}
bind . <Return>   {make_pics; make_hist}
bind . <Double-Button-1> {puts "obs_info: ";eval exec "cat $tmppath/obs_info >@stdout"}
bind . <l>               {puts "obs_info: ";eval exec "cat $tmppath/obs_info >@stdout"}
bind . <i>               {puts "obs_info: ";eval exec "cat $tmppath/obs_info >@stdout"}
bind . <Button-2> {if {[file exists $tmppath/colormap_new]} {eval exec "rm    $tmppath/colormap_new"; eval exec "$callstring &"} \
                   else                                 {eval exec "touch $tmppath/colormap_new"; eval exec "$callstring -colormap new &"};
                   destroy .}
bind . <Button-3> {destroy .}
bind . <q>        {destroy .}
set ct 1
bind . <x> {windowToFile . $ct; incr ct}


if  {$AUTOREFRESH || [file exists /tmp/$UNAME/tcl.pipe]} {
    doforever
}
