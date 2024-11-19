#include "paintscene.h"

paintScene::paintScene(QObject* parent) : QGraphicsScene(parent)
{

}

paintScene::~paintScene()
{

}

void paintScene::mousePressEvent(QGraphicsSceneMouseEvent* event)//drawing ellipse around pressed location
{
    addEllipse(event->scenePos().x() - 14,
        event->scenePos().y() - 14,
        28,
        28,
        QPen(Qt::NoPen),
        QBrush(Qt::white
        ));
    previousPoint = event->scenePos();
}

void paintScene::mouseMoveEvent(QGraphicsSceneMouseEvent* event)//drawing lines based on mouse position
{
    addLine(
        previousPoint.x(),
        previousPoint.y(),
        event->scenePos().x(),
        event->scenePos().y(),
        QPen(Qt::white, 28, Qt::SolidLine, Qt::RoundCap)
    );
    previousPoint = event->scenePos();
}
