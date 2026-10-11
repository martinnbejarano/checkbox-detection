# checkbox-detection

## Demo page

Manual check (`make install && make models` first):

1. `make run` in one terminal and `make frontend-dev` in another; open the URL Vite prints.
2. Drop `samples/sample-003.png`. The page appears with green (checked), red (unchecked) and dashed orange (low confidence) boxes, and the Detections list fills on the right.
3. Hover a row: a blue line joins it to its box and the other boxes fade. Hover a box: its row highlights. Click a row: the page scrolls to the box.
4. Drop a multi-page PDF: the pager switches pages and the list follows.
5. Drop a `.txt` file: the panel shows the error's title and detail.

Low-confidence boxes only appear with `CHECKBOX_LOCATOR=yolox`; the OpenCV locator always scores 1.
